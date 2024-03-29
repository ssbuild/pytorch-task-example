# -*- coding: utf-8 -*-
import json
import logging
import random
import typing

import numpy as np
import torch
from deep_training.data_helper import DataHelper, load_tokenizer, load_configure
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from deep_training.nlp.models.tsdae_model import TransformerForTSDAE, TsdaelArguments
from deep_training.utils.func import seq_pading, seq_padding
from deep_training.utils.trainer import SimpleModelCheckpoint
from fastdatasets.torch_dataset import Dataset as torch_Dataset
from lightning import Trainer
from scipy import stats
from sklearn.metrics.pairwise import paired_distances
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import HfArgumentParser, BertTokenizer

train_info_args = {
    'devices': 1,
    'data_backend': 'record',
    'model_type': 'bert',
    'model_name_or_path': '/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'tokenizer_name': '/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'config_name': '/data/nlp/pre_models/torch/bert/bert-base-chinese/config.json',
    'convert_onnx': False, # 转换onnx模型
    'do_train': True, 
    'do_eval': True,
    # 'train_file': ['/data/nlp/nlp_train_data/clue/afqmc_public/train.json'],
    # 'eval_file': ['/data/nlp/nlp_train_data/clue/afqmc_public/dev.json'],
    # 'test_file': ['/data/nlp/nlp_train_data/clue/afqmc_public/test.json'],
    # 'train_file': ['/data/nlp/nlp_train_data/senteval_cn/LCQMC/LCQMC.train.data'],
    # 'eval_file': ['/data/nlp/nlp_train_data/senteval_cn/LCQMC/LCQMC.valid.data'],
    # 'test_file': ['/data/nlp/nlp_train_data/senteval_cn/LCQMC/LCQMC.test.data'],
    'train_file': ['/data/nlp/nlp_train_data/senteval_cn/STS-B/STS-B.train.data'],
    'eval_file': ['/data/nlp/nlp_train_data/senteval_cn/STS-B/STS-B.valid.data'],
    'test_file': ['/data/nlp/nlp_train_data/senteval_cn/STS-B/STS-B.test.data'],
    # 'train_file': [ '/data/nlp/nlp_train_data/senteval_cn/BQ/BQ.train.data'],
    # 'eval_file': [ '/data/nlp/nlp_train_data/senteval_cn/BQ/BQ.valid.data'],
    # 'test_file': [ '/data/nlp/nlp_train_data/senteval_cn/BQ/BQ.test.data'],
    # 'train_file': ['/data/nlp/nlp_train_data/senteval_cn/ATEC/ATEC.train.data'],
    # 'eval_file': ['/data/nlp/nlp_train_data/senteval_cn/ATEC/ATEC.valid.data'],
    # 'test_file': ['/data/nlp/nlp_train_data/senteval_cn/ATEC/ATEC.test.data'],
    'max_epochs': 5,
    'optimizer': 'adamw',
    'learning_rate': 1e-5,
    'train_batch_size': 10,
    'eval_batch_size': 10,
    'test_batch_size': 2,
    'adam_epsilon': 1e-8,
    'gradient_accumulation_steps': 1,
    'max_grad_norm': 1.0,
    'weight_decay': 0,
    'warmup_steps': 0,
    'output_dir': './output',
    'train_max_seq_length': 64,
    'eval_max_seq_length': 64,
    'test_max_seq_length': 64,
    ##### tsdae 模型参数
    'pooling': 'cls',  # one of [cls,reduce]
    'vector_size': 512,
    'num_encoder_layer': 12,
    'num_decoder_layer': 6,
    'decoder_model_type': 'bert',
    'decoder_model_name_or_path': '/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'decoder_tokenizer_name': '/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'decoder_config_name': '/data/nlp/pre_models/torch/bert/bert-base-chinese/config.json',
}


def pad_to_seqlength(sentence, tokenizer, max_seq_length):
    tokenizer: BertTokenizer
    o = tokenizer(sentence, max_length=max_seq_length, truncation=True, add_special_tokens=True,
                  return_token_type_ids=False)
    arrs = [o['input_ids'], o['attention_mask']]
    seqlen = np.asarray(len(arrs[0]), dtype=np.int64)
    input_ids, attention_mask = seq_pading(arrs, max_seq_length=max_seq_length, pad_val=tokenizer.pad_token_id)
    d = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'seqlen': seqlen
    }
    return d


def add_token_noise(tokens, del_ratio=0.6):
    n = len(tokens)
    if n < 5:
        return tokens
    keep_or_not = np.random.rand(n) > del_ratio
    keep_or_not[0] = True
    keep_or_not[-1] = True
    if sum(keep_or_not) == 0:
        keep_or_not[np.random.randint(1, n - 1, dtype=np.int32)] = True  # guarantee that at least one word remains
    return [tokens[i] for i, bkeep in enumerate(keep_or_not) if bkeep]


class NN_DataHelper(DataHelper):
    # 切分词
    def on_data_process(self, data: typing.Any, mode: str):
        tokenizer: BertTokenizer
        max_seq_length = self.max_seq_length_dict[mode]
        tokenizer = self.tokenizer
        do_lower_case = tokenizer.do_lower_case
        label2id = self.label2id
        decoder_tokenizer = self.decoder_tokenizer


        sentence1, sentence2, label_str = data
        # sentence1, sentence2 independent sample for training
        if mode == 'train':
            d = []
            for sentence in [sentence1, sentence2]:
                tokens_ids = tokenizer.convert_tokens_to_ids(add_token_noise(
                    tokenizer.tokenize(sentence, truncation=True, add_special_tokens=True,
                                       return_token_type_ids=False)))
                seqlen = len(tokens_ids)
                d.append({
                    'input_ids': seq_padding(tokens_ids, max_seq_length=max_seq_length, dtype=np.int32),
                    'attention_mask': seq_padding([1] * seqlen, max_seq_length=max_seq_length, dtype=np.int32),
                    'seqlen': np.asarray(seqlen, dtype=np.int32),
                    **{k + '2': v for k, v in
                       pad_to_seqlength(sentence, decoder_tokenizer, max_seq_length).items()},
                })
        # 评估样本
        else:

            d1 = pad_to_seqlength(sentence1, tokenizer, max_seq_length)
            d2 = pad_to_seqlength(sentence2, tokenizer, max_seq_length)
            d = d1
            for k, v in d2.items():
                d[k + '2'] = v

            if label_str is not None:
                labels = np.asarray(int(label_str), dtype=np.int32)
                d['labels'] = labels
        return d

    # 读取标签
    def on_get_labels(self, files: typing.List[str]):
        return None, None

    # 读取文件
    def on_get_corpus(self, files: typing.List, mode: str):
        D = []
        for filename in files:
            with open(filename, mode='r', encoding='utf-8') as f:
                lines = f.readlines()
                if filename.endswith('.json'):
                    for line in lines:
                        jd = json.loads(line)
                        if not jd:
                            continue
                        D.append((jd['sentence1'], jd['sentence2'], jd.get('label', None)))
                else:
                    for line in lines:
                        line = line.replace('\r\n', '').replace('\n', '')
                        s1, s2, l = line.split('\t', 2)
                        D.append((s1, s2, l))
        # 训练数据重排序
        if mode == 'train':
            tmp = []
            for item in D:
                tmp.append(item[0])
                tmp.append(item[1])
            random.shuffle(tmp)
            D.clear()
            for item1, item2 in zip(tmp[::2], tmp[1::2]):
                D.append((item1, item2, None))

        return D

    def collate_fn(self,batch):
        o = {}
        for i, b in enumerate(batch):
            if i == 0:
                for k in b:
                    o[k] = [torch.tensor(b[k])]
            else:
                for k in b:
                    o[k].append(torch.tensor(b[k]))
        for k in o:
            o[k] = torch.stack(o[k])

        max_len = torch.max(o.pop('seqlen'))
        o['input_ids'] = o['input_ids'][:, :max_len]
        o['attention_mask'] = o['attention_mask'][:, :max_len]

        if 'seqlen2' in o:
            max_len = torch.max(o.pop('seqlen2'))
            o['input_ids2'] = o['input_ids2'][:, :max_len]
            o['attention_mask2'] = o['attention_mask2'][:, :max_len]
        return o


class MyTransformer(TransformerForTSDAE, with_pl=True):
    def __init__(self, *args, **kwargs):
        super(MyTransformer, self).__init__(*args, **kwargs)


def evaluate_sample(a_vecs, b_vecs, labels):
    print('*' * 30, 'evaluating....', a_vecs.shape, b_vecs.shape, labels.shape)
    sims = 1 - paired_distances(a_vecs, b_vecs, metric='cosine')
    print(np.concatenate([sims[:5], sims[-5:]], axis=0))
    print(np.concatenate([labels[:5], labels[-5:]], axis=0))
    correlation, _ = stats.spearmanr(labels, sims)
    print('spearman ', correlation)
    return correlation


class MySimpleModelCheckpoint(SimpleModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super(MySimpleModelCheckpoint, self).__init__(*args, **kwargs)
        self.weight_file = './best.pt'

    def on_save_model(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        pl_module: MyTransformer

        # 当前设备
        device = torch.device('cuda:{}'.format(trainer.global_rank))
        eval_datasets = dataHelper.load_sequential_sampler(dataHelper.eval_files,batch_size=training_args.eval_batch_size,collate_fn=dataHelper.collate_fn)

        a_vecs, b_vecs, labels = [], [], []
        for i, batch in tqdm(enumerate(eval_datasets), total=len(eval_datasets), desc='evalute'):
            for k in batch:
                batch[k] = batch[k].to(device)
            o = pl_module.validation_step(batch, i)
            a_logits, b_logits, b_labels = o['outputs']
            for j in range(len(b_logits)):
                logit1 = np.asarray(a_logits[j], dtype=np.float32)
                logit2 = np.asarray(b_logits[j], dtype=np.float32)
                label = np.asarray(b_labels[j], dtype=np.int32)

                a_vecs.append(logit1)
                b_vecs.append(logit2)
                labels.append(label)

        a_vecs = np.stack(a_vecs, axis=0)
        b_vecs = np.stack(b_vecs, axis=0)
        labels = np.stack(labels, axis=0)
        labels = np.squeeze(labels, axis=-1)

        corrcoef = evaluate_sample(a_vecs, b_vecs, labels)
        f1 = corrcoef
        best_f1 = self.best.get('f1', -np.inf)
        print('current', f1, 'best', best_f1)
        if f1 >= best_f1:
            self.best['f1'] = f1
            logging.info('save best {}, {}\n'.format(self.best['f1'], self.weight_file))
            trainer.save_checkpoint(self.weight_file)


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, TsdaelArguments))
    model_args, training_args, data_args, tsdae_args = parser.parse_dict(train_info_args)

    checkpoint_callback = MySimpleModelCheckpoint(monitor="f1", every_n_epochs=1,
                                                  every_n_train_steps=300 // training_args.gradient_accumulation_steps)
    trainer = Trainer(
        log_every_n_steps=20,
        callbacks=[checkpoint_callback],
        max_epochs=training_args.max_epochs,
        max_steps=training_args.max_steps,
        accelerator="gpu",
        devices=data_args.devices,
        enable_progress_bar=True,
        default_root_dir=data_args.output_dir,
        gradient_clip_val=training_args.max_grad_norm,
        accumulate_grad_batches=training_args.gradient_accumulation_steps,
        num_sanity_val_steps=0,
        strategy='ddp' if torch.cuda.device_count() > 1 else 'auto',
    )

    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer, config, label2id, id2label = dataHelper.load_tokenizer_and_config()
    dataHelper.decoder_tokenizer = None
    dataHelper.decoder_config = None

    decoder_tokenizer, decoder_config = None, None

    #缓存数据
    if data_args.do_train:
        # 加载解码器配置，非训练模式可以不加载
        decoder_tokenizer = load_tokenizer(tokenizer_name=tsdae_args.decoder_tokenizer_name,
                                           model_name_or_path=tsdae_args.decoder_model_name_or_path,
                                           cache_dir=model_args.cache_dir,
                                           do_lower_case=model_args.do_lower_case,
                                           use_fast_tokenizer=model_args.use_fast_tokenizer,
                                           model_revision=model_args.model_revision,
                                           use_auth_token=model_args.use_auth_token,
                                           )

        decoder_config = load_configure(config_name=tsdae_args.decoder_config_name,
                                        model_name_or_path=tsdae_args.decoder_model_name_or_path,
                                        cache_dir=model_args.cache_dir,
                                        model_revision=model_args.model_revision,
                                        use_auth_token=model_args.use_auth_token,
                                        **{
                                            "bos_token_id": tokenizer.bos_token_id,
                                            "pad_token_id": tokenizer.pad_token_id,
                                            "eos_token_id": tokenizer.eos_token_id,
                                            "sep_token_id": tokenizer.sep_token_id
                                        })

        dataHelper.decoder_tokenizer = decoder_tokenizer
        dataHelper.decoder_config = decoder_config


        dataHelper.make_dataset_with_args(data_args.train_file, shuffle=True,mode='train')
    if data_args.do_eval:
        dataHelper.make_dataset_with_args(data_args.eval_file, mode='eval')
    if data_args.do_test:
        dataHelper.make_dataset_with_args(data_args.test_file,mode='test')

    model = MyTransformer(tsdae_args=tsdae_args, decoder_tokenizer=decoder_tokenizer, decoder_config=decoder_config,
                          config=config, model_args=model_args, training_args=training_args)


    if not data_args.convert_onnx:
        train_datasets = dataHelper.load_distributed_random_sampler(
            dataHelper.train_files,
            with_load_memory=True,
            collate_fn=dataHelper.collate_fn,
            batch_size=training_args.train_batch_size,
            num_processes=trainer.world_size, process_index=trainer.global_rank, limit_count=20000)


        if train_datasets is not None:
            trainer.fit(model, train_dataloaders=train_datasets)
        else:
            eval_datasets = dataHelper.load_sequential_sampler(dataHelper.eval_files,batch_size=training_args.eval_batch_size,collate_fn=dataHelper.collate_fn)
            test_datasets = dataHelper.load_sequential_sampler(dataHelper.test_files,batch_size=training_args.test_batch_size,collate_fn=dataHelper.collate_fn)
            if eval_datasets is not None:
                trainer.validate(model, dataloaders=eval_datasets, ckpt_path='./best.pt')

            if test_datasets is not None:
                trainer.test(model, dataloaders=test_datasets, ckpt_path='best.pt')

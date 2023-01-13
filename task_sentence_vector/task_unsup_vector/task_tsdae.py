# -*- coding: utf-8 -*-
import json
import logging
import random
import typing

import numpy as np
import torch
from deep_training.data_helper import DataHelper, load_tokenizer, load_configure
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from deep_training.data_helper import load_tokenizer_and_config_with_args
from deep_training.nlp.models.tsdae_model import TransformerForTSDAE, TsdaelArguments
from deep_training.utils.func import seq_pading, seq_padding
from deep_training.utils.trainer import SimpleModelCheckpoint
from pytorch_lightning import Trainer
from scipy import stats
from sklearn.metrics.pairwise import paired_distances
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import HfArgumentParser, BertTokenizer

train_info_args = {
    'devices':  1,
    'data_backend': 'record',
    'model_type': 'bert',
    'model_name_or_path': '/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'tokenizer_name': '/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'config_name': '/data/nlp/pre_models/torch/bert/bert-base-chinese/config.json',
    'do_train': True,
    'do_eval': True,
     # 'train_file':'/data/nlp/nlp_train_data/clue/afqmc_public/train.json',
    # 'eval_file':'/data/nlp/nlp_train_data/clue/afqmc_public/dev.json',
    # 'test_file':'/data/nlp/nlp_train_data/clue/afqmc_public/test.json',
    'train_file':'/data/nlp/nlp_train_data/senteval_cn/LCQMC/LCQMC.train.data',
    'eval_file':'/data/nlp/nlp_train_data/senteval_cn/LCQMC/LCQMC.valid.data',
    'test_file':'/data/nlp/nlp_train_data/senteval_cn/LCQMC/LCQMC.test.data',
    'max_epochs':3,
    'optimizer': 'adamw',
    'learning_rate':5e-5,
    'train_batch_size': 20,
    'eval_batch_size': 20,
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
    'pooling': 'cls', # one of [cls,reduce]
    'vector_size': 512,
    'num_encoder_layer': 12,
    'num_decoder_layer': 6,
    'decoder_model_type': 'bert',
    'decoder_model_name_or_path': '/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'decoder_tokenizer_name': '/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'decoder_config_name': '/data/nlp/pre_models/torch/bert/bert-base-chinese/config.json',
}

def pad_to_seqlength(sentence,tokenizer,max_seq_length):
    tokenizer: BertTokenizer
    o = tokenizer(sentence, max_length=max_seq_length, truncation=True, add_special_tokens=True,return_token_type_ids=False )
    arrs = [o['input_ids'],o['attention_mask']]
    seqlen = np.asarray(len(arrs[0]),dtype=np.int64)
    input_ids,attention_mask = seq_pading(arrs,max_seq_length=max_seq_length,pad_val=tokenizer.pad_token_id)
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
        keep_or_not[ np.random.randint(1,n-1,dtype=np.int32)] = True # guarantee that at least one word remains
    return [tokens[i] for i,bkeep in enumerate(keep_or_not) if bkeep]

class NN_DataHelper(DataHelper):
    # 切分词
    def on_data_process(self, data: typing.Any, user_data: tuple):
        tokenizer: BertTokenizer
        tokenizer,decoder_tokenizer, max_seq_length, do_lower_case, label2id, mode = user_data
        sentence1, sentence2, label_str = data
        #sentence1, sentence2 independent sample for training
        if mode == 'train':
            d = []
            for sentence in [sentence1,sentence2]:
                tokens_ids = tokenizer.convert_tokens_to_ids(add_token_noise(tokenizer.tokenize(sentence,truncation=True, add_special_tokens=True,return_token_type_ids=False)))
                seqlen = len(tokens_ids)
                d.append({
                    'input_ids': seq_padding(tokens_ids, max_seq_length=max_seq_length, dtype=np.int32),
                    'attention_mask': seq_padding([1] * seqlen, max_seq_length=max_seq_length, dtype=np.int32),
                    'seqlen': np.asarray(seqlen, dtype=np.int32),
                    **{'target_' +k : v for k,v in pad_to_seqlength(sentence,decoder_tokenizer,max_seq_length).items()},
                })
        #评估样本
        else:
            labels = np.asarray(label2id[label_str] if label_str is not None else 0, dtype=np.int64)
            d1 = pad_to_seqlength(sentence1, tokenizer, max_seq_length)
            d2 = pad_to_seqlength(sentence2, tokenizer, max_seq_length)
            d = d1
            for k,v in d2.items():
                d['target_' + k] = v
            d['labels'] = labels
        return d

    # 读取标签
    def on_get_labels(self, files: typing.List[str]):
        D = ['0', '1']
        label2id = {label: i for i, label in enumerate(D)}
        id2label = {i: label for i, label in enumerate(D)}
        return label2id, id2label

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
        return D
    @staticmethod
    def collate_fn(batch):
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

        if 'target_seqlen' in o:
            seqlen = o.pop('target_seqlen')
            max_len = torch.max(seqlen)
            o['target_input_ids'] = o['target_input_ids'][:, :max_len]
            o['target_attention_mask'] = o['target_attention_mask'][:, :max_len]
        return o


class MyTransformer(TransformerForTSDAE, with_pl=True):
    def __init__(self, *args, **kwargs):
        super(MyTransformer, self).__init__(*args, **kwargs)

def evaluate_sample(a_vecs,b_vecs,labels):
    print('*' * 30,'evaluating....',a_vecs.shape,b_vecs.shape,labels.shape)
    sims = 1 - paired_distances(a_vecs,b_vecs,metric='cosine')
    print(np.concatenate([sims[:5] , sims[-5:]],axis=0))
    print(np.concatenate([labels[:5] , labels[-5:]],axis=0))
    correlation,_  = stats.spearmanr(labels,sims)
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
        eval_datasets = dataHelper.load_dataset(dataHelper.eval_files)
        eval_datasets = DataLoader(eval_datasets, batch_size=training_args.eval_batch_size,collate_fn=dataHelper.collate_fn)

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
        labels = np.squeeze(labels,axis=-1)

        corrcoef = evaluate_sample(a_vecs, b_vecs, labels)

        f1 = corrcoef
        best_f1 = self.best.get('f1',-np.inf)
        print('current', f1, 'best', best_f1)
        if f1 >= best_f1:
            self.best['f1'] = f1
            logging.info('save best {}, {}\n'.format(self.best['f1'], self.weight_file))
            trainer.save_checkpoint(self.weight_file)

if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments,TsdaelArguments))
    model_args, training_args, data_args,tsdae_args = parser.parse_dict(train_info_args)

    checkpoint_callback = MySimpleModelCheckpoint(monitor="f1", every_n_train_steps=2000 // training_args.gradient_accumulation_steps)
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
        strategy='ddp' if torch.cuda.device_count() > 1 else None,
    )

    dataHelper = NN_DataHelper(data_args.data_backend)
    tokenizer, config, label2id, id2label = load_tokenizer_and_config_with_args(dataHelper, model_args, training_args,
                                                                                data_args)
    decoder_tokenizer,decoder_config = None,None
    if data_args.do_train:
        #加载解码器配置，非训练模式可以不加载
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
                                    "bos_token_id" : decoder_tokenizer.bos_token_id,
                                    "pad_token_id" : decoder_tokenizer.pad_token_id,
                                    "eos_token_id" : decoder_tokenizer.eos_token_id,
                                    "sep_token_id" : decoder_tokenizer.sep_token_id
                                })


    rng = random.Random(training_args.seed)
    token_fn_args_dict = {
        'train': (tokenizer,decoder_tokenizer, data_args.train_max_seq_length, model_args.do_lower_case, label2id,
                  'train'),
        'eval': (tokenizer,decoder_tokenizer, data_args.eval_max_seq_length, model_args.do_lower_case, label2id,
                 'eval'),
        'test': (tokenizer,decoder_tokenizer, data_args.test_max_seq_length, model_args.do_lower_case, label2id,
                 'test')
    }

    # 缓存数据集
    intermediate_name = data_args.intermediate_name + '_{}'.format(0)
    if data_args.do_train:
        dataHelper.train_files.append(
            dataHelper.make_dataset_with_args(data_args.train_file, token_fn_args_dict['train'],
                                              data_args,
                                              intermediate_name=intermediate_name, shuffle=True,
                                              mode='train'))
    if data_args.do_eval:
        dataHelper.eval_files.append(dataHelper.make_dataset_with_args(data_args.eval_file, token_fn_args_dict['eval'],
                                                                       data_args,
                                                                       intermediate_name=intermediate_name,
                                                                       shuffle=False,
                                                                       mode='eval'))
    if data_args.do_test:
        dataHelper.test_files.append(dataHelper.make_dataset_with_args(data_args.test_file, token_fn_args_dict['test'],
                                                                       data_args,
                                                                       intermediate_name=intermediate_name,
                                                                       shuffle=False,
                                                                       mode='test'))

    train_datasets = dataHelper.load_dataset(dataHelper.train_files, shuffle=True, num_processes=trainer.world_size,
                                             process_index=trainer.global_rank,
                                             infinite=True,
                                             with_record_iterable_dataset=False,
                                             with_load_memory=True)

    if train_datasets is not None:
        train_datasets = DataLoader(train_datasets, batch_size=training_args.train_batch_size,
                                    collate_fn=dataHelper.collate_fn,
                                    shuffle=False if isinstance(train_datasets, IterableDataset) else True)

    model = MyTransformer(tsdae_args=tsdae_args,decoder_tokenizer=decoder_tokenizer,decoder_config=decoder_config,
                          config=config, model_args=model_args, training_args=training_args)

    if train_datasets is not None:
        trainer.fit(model, train_dataloaders=train_datasets)
    else:
        eval_datasets = dataHelper.load_dataset(dataHelper.eval_files)
        test_datasets = dataHelper.load_dataset(dataHelper.test_files)
        if eval_datasets is not None:
            eval_datasets = DataLoader(eval_datasets, batch_size=training_args.eval_batch_size,
                                       collate_fn=dataHelper.collate_fn)
        if test_datasets is not None:
            test_datasets = DataLoader(test_datasets, batch_size=training_args.test_batch_size,
                                       collate_fn=dataHelper.collate_fn)
        if eval_datasets is not None:
            trainer.validate(model, dataloaders=eval_datasets,ckpt_path='./best.pt')

        if test_datasets is not None:
            trainer.test(model, dataloaders=test_datasets,ckpt_path='best.pt')

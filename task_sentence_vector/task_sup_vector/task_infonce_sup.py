# -*- coding: utf-8 -*-
import copy
import json
import logging
import os.path
import typing

import numpy as np
import pytorch_lightning
import torch
from deep_training.data_helper import DataHelper
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from deep_training.nlp.models.infonce import TransformerForInfoNce
from deep_training.utils.trainer import SimpleModelCheckpoint
from pytorch_lightning import Trainer
from scipy import stats
from sklearn.metrics.pairwise import paired_distances
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import HfArgumentParser, BertTokenizer

# model_base_dir = '/data/torch/bert-base-chinese'
model_base_dir = '/data/nlp/pre_models/torch/bert/bert-base-chinese'

train_info_args = {
    'devices': torch.cuda.device_count(),
    'data_backend': 'record',
    'model_type': 'bert',
    'model_name_or_path': model_base_dir,
    'tokenizer_name': model_base_dir,
    'config_name': os.path.join(model_base_dir, 'config.json'),
    'convert_onnx': False, # 转换onnx模型
    'do_train': True, 
    'do_eval': True,
    'do_test': False,
    # 'train_file': ['/data/nlp/nlp_train_data/senteval_cn/LCQMC/LCQMC.train.data'],
    # 'eval_file': ['/data/nlp/nlp_train_data/senteval_cn/LCQMC/LCQMC.valid.data'],
    # 'test_file': ['/data/nlp/nlp_train_data/senteval_cn/LCQMC/LCQMC.test.data'],
    'train_file': [ '/data/nlp/nlp_train_data/senteval_cn/STS-B/STS-B.train.data'],
    'eval_file': [ '/data/nlp/nlp_train_data/senteval_cn/STS-B/STS-B.valid.data'],
    'test_file': [ '/data/nlp/nlp_train_data/senteval_cn/STS-B/STS-B.test.data'],
    # 'train_file': ['/data/nlp/nlp_train_data/senteval_cn/BQ/BQ.train.data'],
    # 'eval_file': ['/data/nlp/nlp_train_data/senteval_cn/BQ/BQ.valid.data'],
    # 'test_file': ['/data/nlp/nlp_train_data/senteval_cn/BQ/BQ.test.data'],
    # 'train_file': ['/data/nlp/nlp_train_data/senteval_cn/ATEC/ATEC.train.data'],
    # 'eval_file': ['/data/nlp/nlp_train_data/senteval_cn/ATEC/ATEC.valid.data'],
    # 'test_file': ['/data/nlp/nlp_train_data/senteval_cn/ATEC/ATEC.test.data'],
    'learning_rate': 3e-5,
    'max_steps': 120000,
    'max_epochs': 1,
    'train_batch_size': 2,
    'eval_batch_size': 10,
    'test_batch_size': 1,
    'adam_epsilon': 1e-8,
    'gradient_accumulation_steps': 20,
    'max_grad_norm': 1.0,
    'weight_decay': 0,
    'warmup_steps': 0,
    'output_dir': './output',
    'train_max_seq_length': 512,
    'eval_max_seq_length': 512,
    'test_max_seq_length': 512,
}

# cls , pooler , last-avg , first-last-avg , reduce
pooling = 'cls'
temperature = 0.1


class NN_DataHelper(DataHelper):
    index = 1

    def on_data_ready(self):
        self.index = -1

    # 切分词
    def on_data_process(self, data: typing.Any, mode: str):
        self.index += 1
        tokenizer: BertTokenizer
        max_seq_length = self.max_seq_length_dict[mode]
        tokenizer = self.tokenizer
        do_lower_case = tokenizer.do_lower_case
        label2id = self.label2id
        # 训练集(sentence1, sentence2, sentence3)  验证集(sentence1, sentence2, labelstr)
        sentence1, sentence2, sentence3_or_labelstr = data
        if mode == 'train':
            o_list = []
            for sentence in [sentence1, sentence2, sentence3_or_labelstr]:
                if sentence is None:  # 无负样本
                    continue
                o = tokenizer(sentence, max_length=max_seq_length, truncation=True, add_special_tokens=True,
                              return_token_type_ids=False)
                for k in o:
                    o[k] = np.asarray(o[k], dtype=np.int32)
                seqlen = np.asarray(len(o['input_ids']), dtype=np.int32)
                pad_len = max_seq_length - seqlen
                if pad_len > 0:
                    pad_val = tokenizer.pad_token_id
                    o['input_ids'] = np.pad(o['input_ids'], pad_width=(0, pad_len), constant_values=(pad_val, pad_val))
                    o['attention_mask'] = np.pad(o['attention_mask'], pad_width=(0, pad_len), constant_values=(0, 0))
                d = {
                    'input_ids': o['input_ids'],
                    'attention_mask': o['attention_mask'],
                    'seqlen': seqlen
                }
                o_list.append(copy.deepcopy(d))

            seqlen = np.max([o.pop('seqlen') for o in o_list])
            d = {k: np.stack([o_list[0][k], o_list[1][k]], axis=0) for k in o_list[0].keys()}
            d['seqlen'] = np.asarray(seqlen, dtype=np.int32)
            return d
        # 验证
        else:
            ds = {}
            for sentence in [sentence1, sentence2]:
                o = tokenizer(sentence, max_length=max_seq_length, truncation=True, add_special_tokens=True,
                              return_token_type_ids=False)
                for k in o:
                    o[k] = np.asarray(o[k], dtype=np.int32)
                seqlen = np.asarray(len(o['input_ids']), dtype=np.int32)
                pad_len = max_seq_length - seqlen
                if pad_len > 0:
                    pad_val = tokenizer.pad_token_id
                    o['input_ids'] = np.pad(o['input_ids'], pad_width=(0, pad_len), constant_values=(pad_val, pad_val))
                    o['attention_mask'] = np.pad(o['attention_mask'], pad_width=(0, pad_len), constant_values=(0, 0))
                d = {
                    'input_ids': o['input_ids'],
                    'attention_mask': o['attention_mask'],
                    'seqlen': seqlen
                }
                if 'input_ids' not in ds:
                    ds = d
                else:
                    for k, v in d.items():
                        ds[k + '2'] = v
            if sentence3_or_labelstr is not None:
                labels = np.asarray(int(sentence3_or_labelstr), dtype=np.int32)
                ds['labels'] = labels
            return ds

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
                        if mode == 'train':
                            if 'sentence3' in jd:
                                D.append((jd['sentence1'], jd['sentence2'], jd['sentence3']))
                            else:
                                D.append((jd['sentence1'], jd['sentence2'], None))
                        else:
                            D.append((jd['sentence1'], jd['sentence2'], jd['label']))
                else:
                    for line in lines:
                        line = line.replace('\r\n', '').replace('\n', '')
                        s3: str
                        s1, s2, s3 = line.split('\t', 2)
                        if mode == 'train':
                            if s3.isdigit() or s3.isdecimal() or s3.isnumeric():
                                D.append((s1, s2, None))
                            else:
                                D.append((s1, s2, s3))
                        else:
                            if s3.isdigit() or s3.isdecimal() or s3.isnumeric():
                                D.append((s1, s2, s3))
                            else:
                                D.append((s1, s2, 1))
                                D.append((s1, s3, 0))

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


def generate_pair_example(all_example_dict: dict):
    all_example_dict = copy.copy(all_example_dict)

    all_example_pos, all_example_neg = [], []
    all_keys = list(all_example_dict.keys())
    np.random.shuffle(all_keys)

    num_all = 0
    for k, v in all_example_dict.items():
        num_all += len(v)
    pos_num_max = num_all // 2 // 5
    for pos_label in all_keys:
        examples = all_example_dict[pos_label]
        if len(examples) == 0:
            continue
        num_size = int(len(examples) // 5) if len(examples) > 100 else np.random.randint(1, min(50, len(examples)),
                                                                                         dtype=np.int32)
        if num_size < 2:
            continue
        id_list = list(range(len(examples)))
        ids = np.random.choice(id_list, replace=False, size=num_size)
        ids = sorted(ids, reverse=True)

        flag = False
        for i1, i2 in zip(ids[::2], ids[1::2]):
            v1 = examples[i1]
            v2 = examples[i2]
            examples.pop(i1)
            examples.pop(i2)
            all_example_pos.append((v1, v2))
            if len(all_example_pos) >= pos_num_max:
                break
        # 去除空标签数据
        if len(examples) <= 1:
            all_keys.remove(pos_label)
        if flag:
            break

    flat_examples = []
    for k in all_keys:
        d_list = all_example_dict[k]
        for d in d_list:
            flat_examples.append((k, d))
    print('construct neg from {} flat_examples'.format(len(flat_examples)))
    idx_list = list(range(len(flat_examples)))
    np.random.shuffle(idx_list)
    while len(idx_list) >= 2:
        flag = False
        k1, e1 = flat_examples[idx_list.pop(0)]
        for i in idx_list[1:]:
            k2, e2 = flat_examples[i]
            if k1 != k2:
                all_example_neg.append((e1, e2))
                idx_list.remove(i)
                if len(all_example_neg) > len(all_example_pos) * 5:
                    flag = True
                    break
                break
        if flag:
            break
    print('pos num', len(all_example_pos), 'neg num', len(all_example_neg))
    return all_example_pos, all_example_neg


def evaluate_sample(a_vecs, b_vecs, labels):
    print('*' * 30, 'evaluating...', a_vecs.shape, b_vecs.shape, labels.shape, 'pos', np.sum(labels))
    sims = 1 - paired_distances(a_vecs, b_vecs, metric='cosine')
    print(np.concatenate([sims[:5], sims[-5:]], axis=0))
    print(np.concatenate([labels[:5], labels[-5:]], axis=0))
    correlation, _ = stats.spearmanr(labels, sims)
    print('spearman ', correlation)
    return correlation


class MyTransformer(TransformerForInfoNce, pytorch_lightning.LightningModule, with_pl=True):
    def __init__(self, *args, **kwargs):
        super(MyTransformer, self).__init__(*args, **kwargs)


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

        corrcoef = evaluate_sample(a_vecs, b_vecs, labels)
        f1 = corrcoef
        best_f1 = self.best.get('f1', -np.inf)
        print('current', f1, 'best', best_f1)
        if f1 >= best_f1:
            self.best['f1'] = f1
            logging.info('save best {}, {}\n'.format(self.best['f1'], self.weight_file))
            trainer.save_checkpoint(self.weight_file)


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
    model_args, training_args, data_args = parser.parse_dict(train_info_args)

    checkpoint_callback = MySimpleModelCheckpoint(every_n_epochs=1,
                                                  every_n_train_steps=500 // training_args.gradient_accumulation_steps)
    trainer = Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=training_args.max_epochs,
        max_steps=training_args.max_steps,
        accelerator="gpu",replace_sampler_ddp=False,
        devices=data_args.devices,
        enable_progress_bar=True,
        default_root_dir=data_args.output_dir,
        gradient_clip_val=training_args.max_grad_norm,
        accumulate_grad_batches=training_args.gradient_accumulation_steps,
        num_sanity_val_steps=0,
        strategy='ddp' if torch.cuda.device_count() > 1 else None,
    )

    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer, config, label2id, id2label = dataHelper.load_tokenizer_and_config()

    # 缓存数据集
    if data_args.do_train:
        dataHelper.make_dataset_with_args(data_args.train_file, shuffle=True,mode='train')
    if data_args.do_eval:
        dataHelper.make_dataset_with_args(data_args.eval_file, mode='eval')
    if data_args.do_test:
        dataHelper.make_dataset_with_args(data_args.test_file,mode='test')

    model = MyTransformer(pooling=pooling, temperature=temperature, config=config, model_args=model_args,
                          training_args=training_args)

    if not data_args.convert_onnx:
        train_datasets = dataHelper.load_random_sampler(dataHelper.train_files,
                                                        with_load_memory=True,
                                                        collate_fn=dataHelper.collate_fn,
                                                        batch_size=training_args.train_batch_size,
                                                        shuffle=True, infinite=True, num_processes=trainer.world_size,
                                                        process_index=trainer.global_rank)

        if train_datasets is not None:
            trainer.fit(model, train_dataloaders=train_datasets)
        else:
            eval_datasets = dataHelper.load_sequential_sampler(dataHelper.eval_files,batch_size=training_args.eval_batch_size,collate_fn=dataHelper.collate_fn)
            test_datasets = dataHelper.load_sequential_sampler(dataHelper.test_files,batch_size=training_args.test_batch_size,collate_fn=dataHelper.collate_fn)

            if eval_datasets is not None:
                trainer.validate(model, dataloaders=eval_datasets, ckpt_path='./best.pt')

            if test_datasets is not None:
                trainer.test(model, dataloaders=test_datasets, ckpt_path='./best.pt')
    else:
        model = MyTransformer.load_from_checkpoint('./best.pt', pooling=pooling, temperature=temperature,
                                                   config=config, model_args=model_args,
                                                   training_args=training_args)
        model.convert_to_onnx('./best.onnx')

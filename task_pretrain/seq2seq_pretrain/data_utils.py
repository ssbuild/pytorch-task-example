# -*- coding: utf-8 -*-
# @Time:  3:12
# @Author:XIE392
# @File：data_utils.py
import json

import numpy as np
import torch
import typing

from deep_training.data_helper import DataHelper, ModelArguments, TrainingArguments, DataArguments
from transformers import BertTokenizer, HfArgumentParser


train_info_args = {
    'devices': 1,
    'data_backend': 'memory_raw',
    'model_type': 't5',
    # 'model_name_or_path': '/data/nlp/pre_models/torch/',
    'tokenizer_name': './t5_small_config',
    'config_name': './t5_small_config/config.json',
    'do_train': True,
    'train_file': [ '/data/nlp/nlp_train_data/thucnews/train.json'],
    'learning_rate': 5e-5,
    'max_epochs': 3,
    'train_batch_size': 10,
    'test_batch_size': 2,
    'adam_epsilon': 1e-8,
    'gradient_accumulation_steps': 1,
    'max_grad_norm': 1.0,
    'weight_decay': 0,
    'warmup_steps': 0,
    'output_dir': './output',
    'train_max_seq_length': 512,
    'eval_max_seq_length': 512,
    'test_max_seq_length': 512,
    'max_target_length': 64,
}

class NN_DataHelper(DataHelper):
    # 切分词
    def on_data_process(self, data: typing.Any, mode: str):
        tokenizer: BertTokenizer
        max_seq_length = self.max_seq_length_dict[mode]
        tokenizer = self.tokenizer

        x = data
        def get_tokenizer_output(text):
            o1 = tokenizer.encode_plus(text, max_length=max_seq_length, truncation=True, add_special_tokens=True, )

            input_ids = np.asarray(o1['input_ids'], dtype=np.int64)
            attention_mask = np.asarray(o1['attention_mask'], dtype=np.int64)
            seqlen = np.asarray(len(input_ids), dtype=np.int64)
            pad_len = max_seq_length - seqlen
            if pad_len > 0:
                pad_val = tokenizer.pad_token_id
                input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
                attention_mask = np.pad(attention_mask, (0, pad_len), 'constant', constant_values=(0, 0))

            out = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'seqlen': seqlen,
            }
            return out

        o1 = get_tokenizer_output(x[0])
        o2 = get_tokenizer_output(x[1])

        d = o1

        d['decoder_input_ids'] = o2['input_ids']
        d['decoder_attention_mask'] = o2['attention_mask']
        d['decoder_seqlen'] = o2['seqlen']

        labels = np.ones_like(d['decoder_input_ids']) * -100
        labels[:-1] = d['decoder_input_ids'][1:]
        return d

    # 读取文件
    def on_get_corpus(self, files: typing.List, mode: str):
        D = []
        for filename in files:
            with open(filename, mode='r', encoding='utf-8') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    jd = json.loads(line)
                    D.append((jd['content'], jd['title']))
        return D

    def collate_fn(self, batch):
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

        max_len = torch.max(o.pop('decoder_seqlen'))
        o['decoder_input_ids'] = o['decoder_input_ids'][:, :max_len]
        o['decoder_attention_mask'] = o['decoder_attention_mask'][:, :max_len]
        o['labels'] = o['labels'][:, :max_len]
        return o


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
    model_args, training_args, data_args = parser.parse_dict(train_info_args)



    dataHelper = NN_DataHelper(data_args.data_backend)
    tokenizer, config, label2id, id2label = dataHelper.load_tokenizer_and_config(model_args, training_args, data_args)


    # 缓存数据集
    if data_args.do_train:
        dataHelper.make_dataset_with_args(data_args.train_file,
                                          data_args, shuffle=True,
                                          mode='train')
    if data_args.do_eval:
        dataHelper.make_dataset_with_args(data_args.eval_file,
                                          data_args,shuffle=False,
                                          mode='eval')
    if data_args.do_test:
        dataHelper.make_dataset_with_args(data_args.test_file,data_args,shuffle=False,mode='test')
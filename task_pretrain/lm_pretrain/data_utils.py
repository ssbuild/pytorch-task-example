# -*- coding: utf-8 -*-
# @Time:  3:02
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
        'model_type': 'gpt2',
        # 预训练模型路径 , 从0训练，则置空
        # 'model_name_or_path': '/data/nlp/pre_models/torch/',
        'tokenizer_name': './gpt2_base_config',
        'config_name': './gpt2_base_config/config.json',
        'is_convert_onnx': False, # 转换onnx模型
    'do_train': True, 
        'train_file': [ '/data/nlp/nlp_train_data/thucnews/train.json'],
        'learning_rate': 5e-5,
        'max_epochs': 3,
        'train_batch_size': 8,
        'test_batch_size': 2,
        'adam_epsilon': 1e-8,
        'gradient_accumulation_steps': 1,
        'max_grad_norm': 1.0,
        'weight_decay': 0,
        'warmup_steps': 0,
        'output_dir': './output',
        'train_max_seq_length': 400,
        'eval_max_seq_length': 512,
        'test_max_seq_length': 512,
    }

class NN_DataHelper(DataHelper):
    # 切分词
    def on_data_process(self, data: typing.Any, mode: str):
        tokenizer: BertTokenizer
        max_seq_length = self.max_seq_length_dict[mode]
        tokenizer = self.tokenizer

        x = data
        if isinstance(x, tuple):
            o = tokenizer(text=x[0], text_pair=x[1], max_length=max_seq_length, truncation=True,
                          add_special_tokens=True)
        else:
            o = tokenizer(x, max_length=max_seq_length, truncation=True, add_special_tokens=True, )

        input_ids = np.asarray(o['input_ids'], dtype=np.int64)
        attention_mask = np.asarray(o['attention_mask'], dtype=np.int64)
        token_type_ids = np.asarray(o['token_type_ids'], dtype=np.int64)

        seqlen = np.asarray(len(input_ids), dtype=np.int64)
        pad_len = max_seq_length - len(input_ids)
        if pad_len > 0:
            pad_val = tokenizer.pad_token_id
            input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            attention_mask = np.pad(attention_mask, (0, pad_len), 'constant', constant_values=(0, 0))
            token_type_ids = np.pad(token_type_ids, (0, pad_len), 'constant', constant_values=(0, 0))
        d = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': np.where(input_ids != tokenizer.pad_token_id, input_ids, np.ones_like(input_ids) * -100),
            'seqlen': seqlen
        }
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
                    if i > 1000:
                        break
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
        if 'token_type_ids' in o:
            o['token_type_ids'] = o['token_type_ids'][:, :max_len]
        o['labels'] = o['labels'][:, :max_len]
        return o

if __name__ == '__main__':


    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
    model_args, training_args, data_args = parser.parse_dict(train_info_args)

    dataHelper = NN_DataHelper(data_args.data_backend)
    tokenizer, config, label2id, id2label = dataHelper.load_tokenizer_and_config(model_args, training_args, data_args)

    # 缓存数据集
    if data_args.do_train:
        dataHelper.make_dataset_with_args(data_args.train_file,data_args, shuffle=True,mode='train')
    if data_args.do_eval:
        dataHelper.make_dataset_with_args(data_args.eval_file,
                                          data_args, shuffle=False,
                                          mode='eval')
    if data_args.do_test:
        dataHelper.make_dataset_with_args(data_args.test_file, data_args, shuffle=False, mode='test')

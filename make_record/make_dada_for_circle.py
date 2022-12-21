# -*- coding: utf-8 -*-
# @Time    : 2022/12/13 8:55

import json
import random
import typing
import numpy as np
import torch
from deep_training.data_helper import DataHelper
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from deep_training.data_helper import load_tokenizer_and_config_with_args
from transformers import HfArgumentParser, BertTokenizer
from fastdatasets import gfile

train_info_args = {
    'devices':  1,
    'data_backend': 'lmdb',
    'model_type': 'bert',
    'model_name_or_path': '/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'tokenizer_name': '/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'config_name': '/data/nlp/pre_models/torch/bert/bert-base-chinese/config.json',
    'do_train': True,
    'do_eval': False,
    'train_file': '/data/nlp/nlp_train_data/lawcup2018/top122/process/*.json',
    'eval_file': '',
    'test_file': '',
    'label_file': '/data/nlp/nlp_train_data/lawcup2018/top122/labels_122.txt',
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
    'max_seq_length': 512
}


class NN_DataHelper(DataHelper):
    # 切分词
    def on_data_process(self, data: typing.Any, user_data: tuple):
        tokenizer: BertTokenizer
        tokenizer, max_seq_length, do_lower_case, label2id, mode = user_data
        sentence, label_str = data

        o = tokenizer(sentence, max_length=max_seq_length, truncation=True, add_special_tokens=True, )
        input_ids = np.asarray(o['input_ids'], dtype=np.int64)
        attention_mask = np.asarray(o['attention_mask'], dtype=np.int64)

        labels = np.asarray(label2id[label_str] if label_str is not None else 0, dtype=np.int64)
        seqlen = np.asarray(len(input_ids), dtype=np.int64)
        pad_len = max_seq_length - len(input_ids)
        if pad_len > 0:
            pad_val = tokenizer.pad_token_id
            input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            attention_mask = np.pad(attention_mask, (0, pad_len), 'constant', constant_values=(0, 0))
        d = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'seqlen': seqlen
        }
        return d

    # 读取标签
    def on_get_labels(self, files: typing.List[str]):
        file = files[0]
        with open(file,mode='r',encoding='utf-8') as f:
            lines = f.readlines()

        labels = []
        for line in lines:
            line = line.replace('\r\n','').replace('\n','')
            if not line:
                continue
            labels.append(line)
        labels = list(set(labels))
        labels = sorted(labels)
        label2id = {l:i for i,l in enumerate(labels)}
        id2label = {i: l for i, l in enumerate(labels)}
        self.label2id = label2id
        self.id2label = id2label
        return self.label2id,self.id2label


    # 读取文件
    def on_get_corpus(self, files: typing.List, mode: str):
        assert len(files) >0
        D = []
        filenames = gfile.glob(files[0])
        for fname in filenames:
            with open(fname, mode='r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    jd = json.loads(line)
                    if not jd:
                        continue
                    D.append((jd['text'], jd.get('label', None)))
        random.shuffle(D)
        return D

    # @staticmethod
    # def collate_fn(batch):
    #     o = {}
    #     for i, b in enumerate(batch):
    #         if i == 0:
    #             for k in b:
    #                 o[k] = [torch.tensor(b[k])]
    #         else:
    #             for k in b:
    #                 o[k].append(torch.tensor(b[k]))
    #     for k in o:
    #         o[k] = torch.stack(o[k])
    #
    #     max_len = torch.max(o.pop('seqlen'))
    #
    #     o['input_ids'] = o['input_ids'][:, :max_len]
    #     o['attention_mask'] = o['attention_mask'][:, :max_len]
    #     if 'token_type_ids' in o:
    #         o['token_type_ids'] = o['token_type_ids'][:, :max_len]
    #     return o


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
    model_args, training_args, data_args = parser.parse_dict(train_info_args)


    dataHelper = NN_DataHelper(data_args.data_backend)
    tokenizer, config, label2id, id2label = load_tokenizer_and_config_with_args(dataHelper, model_args, training_args,
                                                                                data_args)

    token_fn_args_dict = {
        'train': (tokenizer, data_args.train_max_seq_length, model_args.do_lower_case, label2id, 'train'),
        'eval': (tokenizer, data_args.eval_max_seq_length, model_args.do_lower_case, label2id, 'eval'),
        'test': (tokenizer, data_args.test_max_seq_length, model_args.do_lower_case, label2id, 'test')
    }

    N = 1
    train_files, eval_files, test_files = [], [], []
    for i in range(N):
        intermediate_name = data_args.intermediate_name + '_{}'.format(i)
        if data_args.do_train:
            train_files.append(
                dataHelper.make_dataset_with_args(data_args.train_file, token_fn_args_dict['train'], data_args,
                                       intermediate_name=intermediate_name, shuffle=True, mode='train'))
        if data_args.do_eval:
            eval_files.append(
                dataHelper.make_dataset_with_args(data_args.eval_file, token_fn_args_dict['eval'], data_args,
                                       intermediate_name=intermediate_name, shuffle=False, mode='eval'))
        if data_args.do_test:
            test_files.append(
                dataHelper.make_dataset_with_args(data_args.test_file, token_fn_args_dict['test'], data_args,
                                       intermediate_name=intermediate_name, shuffle=False, mode='test'))

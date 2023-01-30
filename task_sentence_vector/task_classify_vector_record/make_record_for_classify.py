# -*- coding: utf-8 -*-
# @Time    : 2022/12/13 8:55

import json
import random
import typing

import numpy as np
from deep_training.data_helper import DataHelper
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from fastdatasets import gfile
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
    'do_eval': False,
    'train_file': gfile.glob('/data/nlp/nlp_train_data/lawcup2018/top122/process/*.json'),
    'eval_file': [ ''],
    'test_file': [ ''],
    'label_file': [ '/data/nlp/nlp_train_data/lawcup2018/top122/labels_122.txt'],
    # 'train_file': [ '/data/nlp/nlp_train_data/clue/tnews/train.json'],
    # 'eval_file': [ ''],
    # 'test_file': [ ''],
    # 'label_file': [ '/data/nlp/nlp_train_data/clue/tnews/labels.txt'],
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
    def on_data_process(self, data: typing.Any, mode: str):
        tokenizer: BertTokenizer
        max_seq_length = self.max_seq_length_dict[mode]
        tokenizer = self.tokenizer
        do_lower_case = tokenizer.do_lower_case
        label2id = self.label2id
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
        with open(file, mode='r', encoding='utf-8') as f:
            lines = f.readlines()

        labels = []
        for line in lines:
            line = line.replace('\r\n', '').replace('\n', '')
            if not line:
                continue
            labels.append(line)
        labels = list(set(labels))
        labels = sorted(labels)
        label2id = {l: i for i, l in enumerate(labels)}
        id2label = {i: l for i, l in enumerate(labels)}
        return label2id, id2label

    # 读取文件
    def on_get_corpus(self, files: typing.List, mode: str):
        assert len(files) > 0
        D = []
        filenames = gfile.glob(files[0])
        for fname in filenames:
            with open(fname, mode='r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    jd = json.loads(line)
                    if not jd:
                        continue
                    if 'text' in jd:
                        text = jd['text']
                    else:
                        text = jd['sentence']
                    D.append((text, jd.get('label', None)))
        random.shuffle(D)
        return D


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
    model_args, training_args, data_args = parser.parse_dict(train_info_args)

    dataHelper = NN_DataHelper(data_args.data_backend)
    tokenizer, config, label2id, id2label = dataHelper.load_tokenizer_and_config(model_args, training_args, data_args)

    if data_args.do_train:
        dataHelper.make_dataset_with_args(data_args.train_file, data_args,
                                              shuffle=True, mode='train')
    if data_args.do_eval:

        dataHelper.make_dataset_with_args(data_args.eval_file, data_args,
                                              shuffle=False, mode='eval')
    if data_args.do_test:
        dataHelper.make_dataset_with_args(data_args.test_file, data_args,
                                              shuffle=False, mode='test')


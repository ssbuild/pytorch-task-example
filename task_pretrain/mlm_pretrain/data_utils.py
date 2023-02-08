# -*- coding: utf-8 -*-
# @Time:  3:09
# @Author:XIE392
# @File：data_utils.py
import json
import random

import torch
import typing
from deep_training.data_helper import DataHelper, ModelArguments, TrainingArguments, MlmDataArguments, DataArguments
from deep_training.utils.maskedlm import make_mlm_wwm_sample
from transformers import BertTokenizer, HfArgumentParser


train_info_args = {
    'devices': 1,
    'data_backend': 'memory_raw',
    'model_type': 'bert',
    'model_name_or_path': '/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'tokenizer_name': '/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'config_name': '/data/nlp/pre_models/torch/bert/bert-base-chinese/config.json',
    'convert_onnx': False, # 转换onnx模型
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
    'do_lower_case': False,
    'do_whole_word_mask': True,
    'max_predictions_per_seq': 20,
    'dupe_factor': 5,
    'masked_lm_prob': 0.15
}


class NN_DataHelper(DataHelper):
    # 切分词
    def on_data_process(self, data: typing.Any, mode: typing.Any):
        tokenizer: BertTokenizer
        max_seq_length = self.max_seq_length_dict[mode]
        tokenizer = self.tokenizer

        rng, do_whole_word_mask, max_predictions_per_seq, masked_lm_prob = self.external_kwargs['mlm_args']

        documents = data
        document_text_string = ''.join(documents)
        document_texts = []
        pos = 0
        while pos < len(document_text_string):
            text = document_text_string[pos:pos + max_seq_length - 2]
            pos += len(text)
            document_texts.append(text)
        # 返回多个文档
        document_nodes = []
        for text in document_texts:
            node = make_mlm_wwm_sample(text, tokenizer, max_seq_length, rng, do_whole_word_mask,
                                       max_predictions_per_seq, masked_lm_prob)
            document_nodes.append(node)
        return document_nodes

    # 读取文件
    def on_get_corpus(self, files: typing.List, mode: str):
        D = []
        line_no = 0
        for input_file in files:
            with open(input_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    jd = json.loads(line)
                    if not jd:
                        continue
                    text = jd['content']
                    docs = text.split('\n\n')
                    D.append([doc for doc in docs if doc])
                    line_no += 1

                    if line_no > 1000:
                        break

                    if line_no % 10000 == 0:
                        print('read_line', line_no)
                        print(D[-1])
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
        if 'token_type_ids' in o:
            o['token_type_ids'] = o['token_type_ids'][:, :max_len]
        o['labels'] = o['labels'][:, :max_len]
        o['weight'] = o['weight'][:, :max_len]
        return o

if __name__ == '__main__':

    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, MlmDataArguments))
    model_args, training_args, data_args, mlm_data_args = parser.parse_dict(train_info_args)

    rng = random.Random(training_args.seed)
    dataHelper = NN_DataHelper(model_args, training_args, data_args, mlm_args=(
    rng, mlm_data_args.do_whole_word_mask, mlm_data_args.max_predictions_per_seq, mlm_data_args.masked_lm_prob))
    tokenizer, config, label2id, id2label = dataHelper.load_tokenizer_and_config()

    # 缓存数据集
    if data_args.do_train:
        dataHelper.make_dataset_with_args(data_args.train_file,
                                          data_args, shuffle=True,
                                          mode='train',
                                          dupe_factor=mlm_data_args.dupe_factor)
    if data_args.do_eval:
        dataHelper.make_dataset_with_args(data_args.eval_file,shuffle=False,mode='eval')
    if data_args.do_test:
        dataHelper.make_dataset_with_args(data_args.test_file,shuffle=False,mode='test')
# -*- coding: utf-8 -*-
# @Time    : 2023/2/13 16:21
import json
import typing

import Levenshtein
import numpy as np
import torch
from deep_training.data_helper import DataHelper, TrainingArguments, DataArguments, ModelArguments
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
    'do_eval': True,
    'train_file': [ '/data/nlp/nlp_train_data/clue/CTC2021/train.json'],
    'eval_file': [ '/data/nlp/nlp_train_data/clue/CTC2021/dev.json'],
    'test_file': [ '/data/nlp/nlp_train_data/clue/CTC2021/test.json'],
    # 'label_file': [ '/data/nlp/nlp_train_data/clue/CTC2021/labels.json'],
    'label_file': [ '/data/nlp/nlp_train_data/clue/CTC2021/vocab.txt'],
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
    'train_max_seq_length': 380,
    'eval_max_seq_length': 512,
    'test_max_seq_length': 512,
}


class NN_DataHelper(DataHelper):
    # 切分词
    def on_data_process(self, data: typing.Any, mode: str):
        tokenizer: BertTokenizer
        max_seq_length = self.max_seq_length_dict[mode]
        tokenizer = self.tokenizer
        do_lower_case = tokenizer.do_lower_case
        label2id = self.label2id

        sentence, label_ops = data

        tokens = list(sentence) if not do_lower_case else list(sentence.lower())
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        tokens = ['[CLS]']  + tokenizer.convert_ids_to_tokens(input_ids) +  ['[SEP]']
        if len(input_ids) > max_seq_length - 2:
            input_ids = input_ids[:max_seq_length - 2]
        input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)

        labels_action = [-100] * max_seq_length
        labels_probs = [-100] * max_seq_length


        for op in label_ops:
            s = op[1] + 1
            e = op[2] + 1

            if e >= max_seq_length:
                print('corpus long length!')
                continue
            for j in range(s,e):
                labels_action[j] = op[0]
                labels_probs[j] = label2id[tokens[j]]

        input_ids = np.asarray(input_ids,np.int32)
        attention_mask = np.asarray(attention_mask, np.int32)
        labels_action = np.asarray(labels_action, np.int32)
        labels_probs = np.asarray(labels_probs, np.int32)

        seqlen = np.asarray(len(input_ids), dtype=np.int64)
        pad_len = max_seq_length - len(input_ids)
        if pad_len > 0:
            pad_val = tokenizer.pad_token_id
            input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            attention_mask = np.pad(attention_mask, (0, pad_len), 'constant', constant_values=(0, 0))
        d = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels_action': labels_action,
            'labels_probs': labels_probs,
            'seqlen': seqlen
        }
        return d

    # 读取标签
    def on_get_labels(self, files: typing.List[str]):
        if files is None:
            return None, None
        label_fname = files[0]
        is_json_file = label_fname.endswith('.json')
        D = set()
        with open(label_fname, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace('\r\n', '').replace('\n', '')
                if not line: continue
                if is_json_file:
                    jd = json.loads(line)
                    line = jd['label']
                D.add(line)
        label2id = {label: i for i, label in enumerate(D)}
        id2label = {i: label for i, label in enumerate(D)}
        return label2id, id2label

    # 读取文件
    def on_get_corpus(self, files: typing.List, mode: str):
        op_map = {
            'equal': 0,
            'insert': 1,
            'delete': 2,
            'replace':3
        }
        D = []
        for filename in files:
            with open(filename, mode='r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    jd = json.loads(line)
                    if not jd:
                        continue
                    src = jd['source']
                    dst = jd.get('target',None)
                    if mode != 'test':
                        assert dst is not None
                    if dst is not None:
                        edits = Levenshtein.opcodes(src, dst)
                        ops = []
                        for item in edits:
                            op = op_map[item[0]]
                            s = item[1]
                            e = item[2]
                            ops.append((op,s,e))
                    else:
                        ops = None
                    D.append((src,ops))
        if mode == 'eval':
            return D[:500]
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

        o['input_ids'] = o['input_ids'][:, :max_len].long()
        o['attention_mask'] = o['attention_mask'][:, :max_len].long()
        if 'token_type_ids' in o:
            o['token_type_ids'] = o['token_type_ids'][:, :max_len].long()

        if 'labels_action' in o:
            o['labels_action'] = o['labels_action'][:, :max_len].long()
            o['labels_probs'] = o['labels_probs'][:, :max_len].long()
        return o



if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
    model_args, training_args, data_args = parser.parse_dict(train_info_args)


    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer, config, label2id, id2label = dataHelper.load_tokenizer_and_config()

    # 缓存数据集
    if data_args.do_train:
        dataHelper.make_dataset_with_args(data_args.train_file, shuffle=True,mode='train')
    if data_args.do_eval:
        dataHelper.make_dataset_with_args(data_args.eval_file, mode='eval')
    if data_args.do_test:
       dataHelper.make_dataset_with_args(data_args.test_file,mode='test')
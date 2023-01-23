# @Time    : 2023/1/22 16:22
# @Author  : tk
# @FileName: data_utils.py
# 数据集 https://github.com/ssbuild/poetry_tang

import copy
import json
import typing

import numpy as np
import torch
from deep_training.data_helper import DataHelper, ModelArguments, TrainingArguments, DataArguments, \
    load_tokenizer_and_config_with_args
from deep_training.utils.func import is_chinese_char
from fastdatasets.record import load_dataset, RECORD
from transformers import BertTokenizer, HfArgumentParser

data_conf = {
    'stride': 50,
    'special': {
        '五绝': '[unused1]',
        '七绝': '[unused2]',
        '五律': '[unused3]',
        '七律': '[unused4]',
    }
}



class NN_DataHelper(DataHelper):
    index = 1

    def on_data_ready(self):
        self.index = -1
    # 切分词
    def on_data_process(self, data: typing.Any, user_data: tuple):
        self.index += 1

        tokenizer: BertTokenizer
        tokenizer, max_seq_length, do_lower_case, label2id, mode = user_data
        sub_list = data


        input_ids = []
        token_type_ids = []
        #每1千首
        for idx, (type, title, paragraphs) in enumerate(sub_list):
            if type is None:
                type = ''
            o = tokenizer.encode_plus(text= type + title ,text_pair=''.join(paragraphs), max_length=max_seq_length, truncation=True,return_attention_mask=False,add_special_tokens=False)
            input_ids += o['input_ids']
            token_type_ids += o['token_type_ids']
            if idx != len(sub_list) - 1:
                input_ids += [tokenizer.sep_token_id]
                token_type_ids += [1]

        stride = data_conf['stride']

        pos = 0
        ds = []
        while pos < len(input_ids):
            input_ids_ = [tokenizer.cls_token_id] + input_ids[pos: pos + max_seq_length -2] + [tokenizer.sep_token_id]
            attention_mask_ = [1] * len(input_ids_)
            token_type_ids_ = [0] + token_type_ids[pos: pos + max_seq_length -2] + [1]
            pos += stride
            seqlen = np.asarray(len(input_ids_),dtype=np.int32)
            pad_len = max_seq_length - seqlen
            input_ids_=  np.asarray(input_ids_,dtype=np.int32)
            attention_mask_ = np.asarray(attention_mask_, dtype=np.int32)
            token_type_ids_ = np.asarray(token_type_ids_, dtype=np.int32)
            if pad_len:
                pad_val = tokenizer.pad_token_id
                input_ids_ = np.pad(input_ids_, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
                attention_mask_ = np.pad(attention_mask_, (0, pad_len), 'constant', constant_values=(0, 0))
                token_type_ids_ = np.pad(token_type_ids_, (0, pad_len), 'constant', constant_values=(0, 0))
            d = {
                'input_ids': input_ids_,
                'attention_mask': attention_mask_,
                'token_type_ids': token_type_ids_,
                'seqlen': seqlen
            }
            ds.append(d)
        if self.index < 3:
            print(ds[0])
        return ds


    # 读取文件
    def on_get_corpus(self, files:typing.List, mode:str):
        D = []
        dataset = load_dataset.RandomDataset(files, options=RECORD.TFRecordOptions(compression_type='GZIP')).parse_from_numpy_writer()

        def poetry_parser(x):
            x = str(x['node'].tolist(), encoding='utf-8')
            x = json.loads(x)
            return x
        dataset = dataset.map(poetry_parser)
        #单条数据
        #{'author': '徐铉', 'title': '春尽日游后湖赠刘起居', 'paragraphs': ['今朝湖上送春归，万顷澄波照白髭。', '笑折残花劝君酒，金丹成熟是何时。'], 'tones': ['平平平仄仄平平，仄仄平平仄仄平。', '仄仄平平仄平仄，平平平仄仄平平。']}
        sub = []


        def is_format(paragraphs: typing.List[typing.AnyStr]):
            length = 0
            flag = True
            for idx,sentence in enumerate(paragraphs):
                n = 0
                for char in sentence:
                    if is_chinese_char(ord(char)):
                        n += 1
                if idx == 0:
                    length = n
                    continue
                if n != length:
                    flag = True

            return flag

        special = data_conf['special']
        for i in range(len(dataset)):
            d = dataset[i]
            title = d['title']
            paragraphs = d['paragraphs']
            tones = d['tones']
            type = None
            if is_format(paragraphs):
                if len(paragraphs) == 2:
                    if paragraphs[0].find('，'):
                        length = len(paragraphs[0].split('，')[0])
                        if length == 5:
                            type = special['五绝']
                        elif length == 7:
                            type = special['七绝']
                elif len(paragraphs) == 4:
                    if paragraphs[0].find('，'):
                        length = len(paragraphs[0].split('，')[0])
                        if length == 5:
                            type = special['五律']
                        elif length == 7:
                            type = special['七律']
            # 每1千首为一组
            if len(sub) < 1000:
                sub.append((type,title,paragraphs))
            else:
                D.append(copy.deepcopy(sub))
                sub.clear()
        if sub:
            D.append(copy.deepcopy(sub))
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
        o['token_type_ids'] = o['token_type_ids'][:, :max_len]
        o['labels'] = torch.clone(o['input_ids']).long()
        return o


if __name__ == '__main__':
    train_info_args = {
        'devices': 1,
        'data_backend': 'record',
        'model_type': 'bert',
        'model_name_or_path': '/data/nlp/pre_models/torch/bert/bert-base-chinese',
        'tokenizer_name': '/data/nlp/pre_models/torch/bert/bert-base-chinese',
        'config_name': '/data/nlp/pre_models/torch/bert/bert-base-chinese/config.json',
        'do_train': True,
        'train_file': '/data/nlp/nlp_train_data/poetry/tangsong.record',
        'output_dir': './output',
        'max_seq_length': 512,
    }

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
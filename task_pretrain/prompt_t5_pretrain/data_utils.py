# @Time    : 2023/1/22 16:22
# @Author  : tk
# @FileName: data_utils.py
#reference: https://github.com/clue-ai/PromptCLUE/blob/main/Fine_tuning_PyTorch.ipynb

import copy
import json
import os
import random
import typing

import numpy as np
import torch
from deep_training.data_helper import DataHelper, ModelArguments, TrainingArguments, DataArguments
from deep_training.utils.func import is_chinese_char
from fastdatasets.record import load_dataset as Loader, RECORD, WriterObject, gfile
from tqdm import tqdm
from transformers import BertTokenizer, HfArgumentParser

train_info_args = {
    'devices': 1,
    'data_backend': 'record',
    'model_type': 't5',
    # 预训练模型路径 , 从0训练，则置空
    # 'model_name_or_path': '/data/nlp/pre_models/torch/',
    'tokenizer_name': './t5_small_config',
    'config_name': './t5_small_config/config.json',
    'convert_onnx': False, # 转换onnx模型
    'do_train': True, 
    'train_file':  [ '/data/nlp/nlp_train_data/clueprompt/finetune_train_examples.json'],
    'max_epochs': 3,
    'train_batch_size': 10,
    'eval_batch_size': 2,
    'test_batch_size': 2,
    'learning_rate': 5e-5,
    'adam_epsilon': 1e-8,
    'gradient_accumulation_steps': 1,
    'max_grad_norm': 1.0,
    'weight_decay': 0,
    'warmup_steps': 0,
    'output_dir': './output',
    'max_seq_length': 512,
    'max_target_length': 100  # 预测最大长度
}



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

        doc_type,src_text,tgt_text = data

        o1 = tokenizer.encode_plus(text=src_text, truncation=True,max_length=max_seq_length)
        o2 = tokenizer.encode_plus(text=tgt_text, truncation=True,max_length=max_seq_length)

        input_ids = np.asarray(o1['input_ids'], dtype=np.int64)
        attention_mask = np.asarray(o1['attention_mask'], dtype=np.int64)
        seqlen = np.asarray(len(input_ids), dtype=np.int64)
        pad_len = max_seq_length - seqlen
        if pad_len > 0:
            pad_val = tokenizer.pad_token_id
            input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            attention_mask = np.pad(attention_mask, (0, pad_len), 'constant', constant_values=(0, 0))

        decoder_input_ids = np.asarray(o2['input_ids'], dtype=np.int64)
        decoder_attention_mask = np.asarray(o2['attention_mask'], dtype=np.int64)
        labels = np.asarray(o2['input_ids'][1:], dtype=np.int64)
        decoder_seqlen = np.asarray(len(decoder_input_ids), dtype=np.int64)
        pad_len = max_seq_length - decoder_seqlen
        if pad_len > 0:
            pad_val = tokenizer.pad_token_id
            decoder_input_ids = np.pad(decoder_input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            decoder_attention_mask = np.pad(decoder_attention_mask, (0, pad_len), 'constant', constant_values=(0, 0))
            labels = np.pad(labels, (0, pad_len+1), 'constant', constant_values=(-100, -100))

        d = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'seqlen': seqlen,
            'decoder_input_ids': decoder_input_ids,
            'decoder_attention_mask': decoder_attention_mask,
            'decoder_seqlen': decoder_seqlen,
            'labels':labels
        }
        return d

    # 读取文件
    def on_get_corpus(self, files: typing.List, mode: str):
        D = []
        #{"input": "我可以用以下的句子：“花呗在什么时间段可以用”，来替换这个句子：“什么时候用了花贝”，并且它们有相同的意思？。选项：是的，不是。答案：", "target": "不是", "type": "classify"}
        for file in files:
            with open(file,mode='r',encoding='utf-8',newline='\n') as f:
                lines = f.readlines()

            for line in lines:
                jd = json.loads(line)
                if not jd:
                    continue
                doc_type = jd.get('type', '')
                src_text = jd['input']
                tgt_text = jd['target']
                D.append((doc_type,src_text,tgt_text))
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

        max_len = torch.max(o.pop('seqlen')).numpy().tolist()
        decoder_seqlen = torch.max(o.pop('decoder_seqlen')).numpy().tolist()


        o['input_ids'] = o['input_ids'][:, :max_len]
        o['attention_mask'] = o['attention_mask'][:, :max_len]
        o['decoder_input_ids'] = o['decoder_input_ids'][:, :decoder_seqlen]
        o['decoder_attention_mask'] =  o['decoder_attention_mask'][:, :decoder_seqlen]
        o['labels'] =  o['labels'][:, :decoder_seqlen]
        return o


if __name__ == '__main__':

    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
    model_args, training_args, data_args = parser.parse_dict(train_info_args)

    dataHelper = NN_DataHelper(data_args.data_backend)
    tokenizer, config, label2id, id2label = dataHelper.load_tokenizer_and_config(model_args, training_args, data_args)
    config.decoder_start_token_id = tokenizer.cls_token_id
    # 缓存数据集
    # 检测是否存在 output/dataset_0-train.record ，不存在则制作数据集
    if data_args.do_train:
        dataHelper.make_dataset_with_args(data_args.train_file,data_args, shuffle=True,mode='train')
    if data_args.do_eval:
        dataHelper.make_dataset_with_args(data_args.eval_file,
                                           data_args, shuffle=False,
                                           mode='eval')
    if data_args.do_test:
        dataHelper.make_dataset_with_args(data_args.test_file,data_args, shuffle=False,mode='test')


    def shuffle_records(record_filenames, outfile, compression_type='GZIP'):
        print('shuffle_records record...')
        options = RECORD.TFRecordOptions(compression_type=compression_type)
        dataset_reader = Loader.RandomDataset(record_filenames, options=options, with_share_memory=True)
        data_size = len(dataset_reader)
        all_example = []
        for i in tqdm(range(data_size), desc='load records'):
            serialized = dataset_reader[i]
            all_example.append(serialized)
        dataset_reader.close()

        shuffle_idx = list(range(data_size))
        random.shuffle(shuffle_idx)
        writer = WriterObject(outfile, options=options)
        for i in tqdm(shuffle_idx, desc='shuffle record'):
            example = all_example[i]
            writer.write(example)
        writer.close()


    # 再次打乱数据
    shuffle_records(dataHelper.train_files, dataHelper.train_files[0])

# -*- coding: utf-8 -*-
# @Time    : 2023/1/30 11:18

import os
import random
import numpy as np
from fastdatasets.record import load_dataset as Loader, gfile, RECORD, WriterObject
from tqdm import tqdm
from transformers import BertTokenizer



path_list = [
    '/data/nlp/pre_models/torch/bert/bert-base-chinese',
    '/data/torch/bert-base-chinese',
    '/opt/tk/torch/bert-base-chinese'
]
path = ''
for p in path_list:
    if os.path.exists(p):
        path = p
        break

tokenizer = BertTokenizer.from_pretrained(path)

# 拆分数据集
def load_record(input_record_filenames,  compression_type='GZIP'):
    print('load_record record...')
    options = RECORD.TFRecordOptions(compression_type=compression_type)
    dataset_reader = Loader.RandomDataset(input_record_filenames, options=options, with_share_memory=True)
    dataset_reader = dataset_reader.parse_from_numpy_writer()

    for i in tqdm(range(len(dataset_reader)), desc='load records'):
        exampe = dataset_reader[i]

        print(exampe.keys())
        seqlen = exampe['seqlen']
        seqlen = np.squeeze(seqlen,axis=-1)
        input_ids = exampe['input_ids']
        input_ids = input_ids[:seqlen]
        tokens = tokenizer.decode(input_ids)
        print(''.join(tokens))
        if i > 10:
            break
    dataset_reader.close()

print('*' * 30)

load_record('./train.record')


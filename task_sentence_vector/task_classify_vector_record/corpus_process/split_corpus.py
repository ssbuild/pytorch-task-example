# -*- coding: utf-8 -*-
# @Time    : 2023/1/30 15:32
import json
import random
from tqdm import tqdm
import numpy as np

np.random.seed(123456)

shuffle_idx = None

def process_file(in_file,train_file,eval_file):
    global shuffle_idx
    with open(in_file,mode='r',encoding='utf-8') as f:
        lines = f.readlines()

    f1 = open(train_file, mode='w', encoding='utf-8', newline='\n')
    f2 = open(eval_file,  mode='w', encoding='utf-8', newline='\n')

    if shuffle_idx is None:
        shuffle_idx = list(range(len(lines)))
        np.random.shuffle(shuffle_idx)
    else:
        if len(lines) != len(shuffle_idx):
            raise ValueError('NOT EQ')
    print(shuffle_idx[:100])
    for i,idx in tqdm(enumerate(shuffle_idx),total=len(lines)):
        jd = json.loads(lines[idx])
        if i % 15 == 0:
            f = f2
        else:
            f = f1
        f.write(json.dumps(jd, ensure_ascii=False) + '\n')
    f1.close()
    f2.close()




if __name__ == '__main__':
    in_file = '/data/nlp/nlp_train_data/lawcup2018/top122/jieba_process_output/jieba_process.json'
    train_file =  '/data/nlp/nlp_train_data/lawcup2018/top122/jieba_process_output/train_jieba.json'
    eval_file = '/data/nlp/nlp_train_data/lawcup2018/top122/jieba_process_output/eval_jieba.json'

    process_file(in_file,train_file,eval_file)

    in_file = '/data/nlp/nlp_train_data/lawcup2018/top122/jieba_process_output/raw.json'
    train_file = '/data/nlp/nlp_train_data/lawcup2018/top122/jieba_process_output/train.json'
    eval_file = '/data/nlp/nlp_train_data/lawcup2018/top122/jieba_process_output/eval.json'

    process_file(in_file, train_file, eval_file)
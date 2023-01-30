# @Time    : 2023/1/9 23:26
# @Author  : tk
# @FileName: stopwards.py
import json
import jieba
from tqdm import tqdm
import re
from collections import Counter
import os


def get_cipin(fs,outdir,stopwards_file='./stopwards.txt'):
    stopwards = set()
    with open(stopwards_file, mode='r', encoding='utf-8', newline='\n') as f:
        while True:
            text = f.readline()
            if not text:
                break
            text = text.strip('\r\n').strip('\n')
            stopwards.add(text)

    print(list(stopwards)[:100])
    counter = Counter()
    f_out = open(os.path.join(outdir, 'jieba_process.json'), mode='w', encoding='utf-8', newline='\n')
    f_out2 = open(os.path.join(outdir, 'raw.json'), mode='w', encoding='utf-8', newline='\n')
    for filename in tqdm(fs,total=len(fs)):
        with open(filename,mode='r',encoding='utf-8') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                jd = json.loads(line)
                if not jd:
                    continue
                text = jd['text']
                label = jd['label']
                text = text.strip('\n')
                text = re.sub("[A-Za-z0-9\：\·\—\，\。\“ \”]", "", text)
                seg_list = jieba.cut(text,cut_all=False)

                seg_list_new = [s for s in seg_list if s not in stopwards]
                counter.update(seg_list_new)

                o = {
                    'text': ' '.join(seg_list_new),
                    'label': label
                }
                f_out.write(json.dumps(o,ensure_ascii=False) + '\n')
                f_out2.write(json.dumps(jd, ensure_ascii=False) + '\n')
    f_out.close()
    f_out2.close()

    print('\n词频统计结果：')
    vocabfile = os.path.join(outdir, 'vocab.txt')
    with open(vocabfile,mode='w',encoding='utf-8',newline='\n') as f:
        for (k,v) in counter.most_common():# 输出词频最高的前两个词
            print("%s:%d"%(k,v))
            f.write("{} {}\n".format(k,v))


if __name__ == '__main__':
    data_dir = '/data/nlp/nlp_train_data/lawcup2018/top122/process'
    outdir='/data/nlp/nlp_train_data/lawcup2018/top122/jieba_process_output'

    fs = os.listdir(data_dir)
    get_cipin([os.path.join(data_dir,f) for f in fs],outdir)
# -*- coding: utf-8 -*-
# @Time    : 2023/1/10 16:55
import copy
import os
import random

import numpy as np
from fastdatasets.record import load_dataset as Loader, gfile, RECORD, NumpyWriter
from tqdm import tqdm


def generate_pair_example(all_example_dict: dict):
    all_example_pos,all_example_neg = [],[]
    all_keys = list(all_example_dict.keys())
    np.random.shuffle(all_keys)
    all_example_num = {lable: list(range(len(all_example_dict[lable]))) for lable in all_example_dict}
    for pos_label in all_keys:
        examples = all_example_dict[pos_label]
        idx_list: list
        idx_list_negs: list
        idx_list = all_example_num[pos_label]
        if len(idx_list) == 0:
            continue

        num_size = int(min(np.random.randint(300,1000), int(len(idx_list) * 0.5)))
        if num_size < 2:
            continue

        idx = np.random.choice(idx_list, replace=False, size=num_size)
        for i1,i2 in zip(idx[::2],idx[1::2]):
            v1 = examples[i1]
            v2 = examples[i2]
            idx_list.remove(i1)
            idx_list.remove(i2)
            all_example_pos.append((v1, v2))

        # 去除空标签数据
        if len(idx_list) <= 1:
            all_keys.remove(pos_label)
    all_example = []
    for k,d_list in all_example_dict.items():
        for d in d_list:
            all_example.append((k,d))

    idx_list = list(range(len(all_example)))
    np.random.shuffle(idx_list)

    while len(idx_list) > 1:
        flag = False
        k1,e1 = all_example[idx_list.pop(0)]
        for i in idx_list[1:]:
            k2,e2 = all_example[i]
            if k1 != k2:
                all_example_neg.append((e1,e2))
                idx_list.remove(i)
                if len(all_example_neg) > 2* len(all_example_pos):
                    flag = True
                    break
        if flag:
            break
    print('pos num',len(all_example_pos),'neg num',len(all_example_neg) )
    return all_example_pos,all_example_neg

def make_pos_neg_records(input_record_filenames, output_file, compression_type='GZIP'):
    print('make_pos_neg_records record...')
    options = RECORD.TFRecordOptions(compression_type=compression_type)
    dataset_reader = Loader.RandomDataset(input_record_filenames, options=options, with_share_memory=True).parse_from_numpy_writer()
    data_size = len(dataset_reader)
    all_example_dict = {}

    for i in tqdm(range(data_size), desc='load records'):
        serialized = dataset_reader[i]
        labels = serialized['labels']
        labels = np.squeeze(labels).tolist()
        if labels not in all_example_dict:
            all_example_dict[labels] = []
        all_example_dict[labels].append(serialized)

    if hasattr(dataset_reader,'close'):
        dataset_reader.close()
    else:
        dataset_reader.reset()

    print(all_example_dict.keys())
    all_example_pos,all_example_neg = generate_pair_example(all_example_dict)


    writer = NumpyWriter(output_file, options=options)
    num_list = [0,0]
    obj_list = [(all_example_pos,1),(all_example_neg,0)]
    for n,(data_list,sim) in enumerate(obj_list):
        shuffle_idx = list(range(len(data_list)))
        for i in tqdm(shuffle_idx, desc='write record',total=len(shuffle_idx)):
            num_list[n] += 1
            example1,example2 = data_list[i]
            example_new = example1
            example_new['input_ids2'] = example2['input_ids']
            example_new['attention_mask2'] = example2['attention_mask']
            example_new['labels2'] = example2['labels']
            example_new['seqlen2'] = example2['seqlen']
            example_new['sim'] = np.asarray(sim,dtype=np.int32)
            writer.write(example_new)
    writer.close()

    print('nums',num_list)


if __name__ == '__main__':
    example_files = r'/data/record/cse_0110/eval.record'

    output_train_file = os.path.join('/data/record/cse_0110/eval_pos_neg.record')

    make_pos_neg_records(input_record_filenames=example_files,
                         output_file=output_train_file, )
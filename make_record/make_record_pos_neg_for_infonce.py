# @Time    : 2022/12/16 11:03
# @Author  : tk
# @FileName: split_record.py
import copy
import os
import random

import numpy as np
from fastdatasets.record import load_dataset as Loader, gfile, RECORD, NumpyWriter
from tqdm import tqdm


def gen_pos_neg_records(all_example):
    all_example_new = []
    all_keys = list(all_example.keys())
    all_example_num = {lable: list(range(len(all_example[lable]))) for lable in all_example}

    for lable in tqdm(all_example.keys()):
        all_keys.remove(lable)
        examples = all_example[lable]
        idx_list: list
        idx_list_negs: list
        idx_list = all_example_num[lable]



        while len(idx_list):
            one_sample_pos, one_sample_neg = [], []
            idx = np.random.choice(idx_list, replace=False,size=min(50, len(idx_list)))
            for value in idx:
                idx_list.remove(value)
                one_sample_pos.append(examples[value])

            if len(one_sample_pos) < 2:
                continue

            keys = np.random.choice(all_keys,replace=False, size=min(40,len(all_keys)))
            keys = list(set(keys))

            for key in keys:
                examples_negs = all_example[key]
                idx_list_negs = all_example_num[key]
                if len(idx_list_negs) == 0:
                    continue
                ids = np.random.choice(idx_list_negs,replace=False, size=min(25,len(idx_list_negs)))
                for value in ids:
                    if random.random() < 0.5:
                        idx_list_negs.remove(value)
                    one_sample_neg.append(examples_negs[value])

            if len(one_sample_neg) < 30:
                continue

            all_example_new.append((one_sample_pos, one_sample_neg))

            if len(all_example_new) % 10000 == 0:
                print('current num',len(all_example_new))


    return all_example_new

def make_pos_neg_records(input_record_filenames, output_file, compression_type='GZIP'):
    print('make_pos_neg_records record...')
    options = RECORD.TFRecordOptions(compression_type=compression_type)
    dataset_reader = Loader.RandomDataset(input_record_filenames, options=options, with_share_memory=True).parse_from_numpy_writer()
    data_size = len(dataset_reader)
    all_example = {}

    for i in tqdm(range(data_size), desc='load records'):
        serialized = dataset_reader[i]
        labels = serialized['labels']
        labels = np.squeeze(labels).tolist()
        if labels not in all_example:
            all_example[labels] = []
        all_example[labels].append(serialized)


    if hasattr(dataset_reader,'close'):
        dataset_reader.close()
    else:
        dataset_reader.reset()

    print(all_example.keys())
    all_example_new = gen_pos_neg_records(all_example)



    print('all_example_new',len(all_example_new))


    writer = NumpyWriter(output_file, options=options)

    shuffle_idx = list(range(len(all_example_new)))
    random.shuffle(shuffle_idx)

    num_train = 0
    total_n = 0
    for i in tqdm(shuffle_idx, desc='shuffle record',total=len(shuffle_idx)):
        example = all_example_new[i]
        num_train += 1
        example_new = {}
        pos,neg = example
        total_n += len(pos) + len(neg)

        example_new['pos_len'] = np.asarray(len(pos),dtype=np.int32)
        example_new['neg_len'] = np.asarray(len(neg), dtype=np.int32)
        d: dict
        for idx,d in enumerate(pos):
            example_new['input_ids_pos{}'.format(idx)]= d['input_ids']
            example_new['attention_mask_pos{}'.format(idx)] = d['attention_mask']
            example_new['labels_pos{}'.format(idx)] = d['labels']
            example_new['seqlen_pos{}'.format(idx)] = d['seqlen']

        for idx, d in enumerate(neg):
            example_new['input_ids_neg{}'.format(idx)] = d['input_ids']
            example_new['attention_mask_neg{}'.format(idx)] = d['attention_mask']
            example_new['labels_neg{}'.format(idx)] = d['labels']
            example_new['seqlen_neg{}'.format(idx)] = d['seqlen']

        writer.write(example_new)
    writer.close()
    print('num train record',num_train,'total record',total_n)


if __name__ == '__main__':



    example_files = r'/home/tk/train/make_big_data/pos_neg/train.record'

    output_train_file = os.path.join('/home/tk/train/make_big_data/pos_neg/train_pos_neg.record')



    make_pos_neg_records(input_record_filenames=example_files,
                         output_file=output_train_file, )
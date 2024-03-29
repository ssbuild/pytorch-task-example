# @Time    : 2022/12/16 11:03
# @Author  : tk
# @FileName: split_record.py

import os
import random

from fastdatasets.record import load_dataset as Loader, gfile, RECORD, WriterObject
from tfrecords import TFRecordOptions
from tqdm import tqdm


# 拆分数据集
def split_records(input_record_filenames, output_train_file, output_eval_file, compression_type='GZIP'):
    print('split_records record...')
    options = RECORD.TFRecordOptions(compression_type=compression_type)
    dataset_reader = Loader.RandomDataset(input_record_filenames, options=options, with_share_memory=True)

    all_example = []
    for i in tqdm(range(len(dataset_reader)), desc='load records'):
        serialized = dataset_reader[i]
        all_example.append(serialized)
    dataset_reader.close()

    # #小样本
    # all_example = all_example[:10000]
    data_size = len(all_example)
    shuffle_idx = list(range(data_size))
    random.shuffle(shuffle_idx)

    writer_train = WriterObject(output_train_file, options=TFRecordOptions(compression_type='GZIP'))
    writer_eval = WriterObject(output_eval_file, options=TFRecordOptions(compression_type='GZIP'))

    num_train = 0
    num_eval = 0
    for i in tqdm(shuffle_idx, desc='shuffle record'):
        example = all_example[i]

        if (i + 1) % 15 == 0:
            num_eval += 1
            writer = writer_eval
        else:
            num_train += 1
            writer = writer_train

        writer.write(example)

    writer_train.close()
    writer_eval.close()

    print('num_train', num_train, 'num_eval', num_eval)


if __name__ == '__main__':
    src_files = [
        '/data/record/cse/dataset_0-train.record'
    ]
    dst_dir = '/data/record/cse_1226/'

    if not os.path.exists(dst_dir):
        gfile.makedirs(dst_dir)

    output_train_file = os.path.join(dst_dir, 'train.record')
    output_eval_file = os.path.join(dst_dir, 'eval.record')

    split_records(input_record_filenames=src_files,
                  output_train_file=output_train_file,
                  output_eval_file=output_eval_file)

# @Time    : 2022/12/16 10:58
# @Author  : tk
# @FileName: shuffle_record.py

import os
import random

from fastdatasets.record import load_dataset as Loader, RECORD, WriterObject
from tqdm import tqdm


def shuffle_records(record_filenames, out_dir, out_record_num, compression_type='GZIP'):
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
    writers = [WriterObject(os.path.join(out_dir, 'record_gzip_shuffle_{}.record'.format(i)), options=options) for i in
               range(out_record_num)]
    for i in tqdm(shuffle_idx, desc='shuffle record'):
        example = all_example[i]
        writers[i % out_record_num].write(example)
    for writer in writers:
        writer.close()


if __name__ == '__main__':
    src_records = ['/tmp/train.record']
    dst_dir = '/tmp/'
    shuffle_records(record_filenames=src_records, out_dir=dst_dir, out_record_num=1)

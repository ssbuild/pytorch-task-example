# @Time    : 2022/12/16 11:03
# @Author  : tk
# @FileName: split_record.py

import os

from fastdatasets.record import load_dataset as Loader, gfile, RECORD, WriterObject
from tfrecords import TFRecordOptions
from tqdm import tqdm


# 合并数据集
def merge_records(input_record_filenames, output_file, compression_type='GZIP'):
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
    writer_output = WriterObject(output_file, options=TFRecordOptions(compression_type='GZIP'))

    for i in tqdm(shuffle_idx, desc='write record'):
        example = all_example[i]
        writer_output.write(example)
    writer_output.close()

    print('num', len(shuffle_idx))


if __name__ == '__main__':
    src_files = [
        '/data/record/cse_0110/eval_pos.record.cache',
        '/data/record/cse_0110/eval_neg.record.cache'
    ]
    dst_dir = './'
    if not os.path.exists(dst_dir):
        gfile.makedirs(dst_dir)

    output_file = os.path.join(dst_dir, 'eval_pos_neg.record.cache')

    merge_records(input_record_filenames=src_files,
                  output_file=output_file, )

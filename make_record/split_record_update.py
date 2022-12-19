# @Time    : 2022/12/16 11:03
# @Author  : tk
# @FileName: split_record.py

import os

import numpy as np
from fastdatasets.record import load_dataset as Loader, gfile, RECORD, NumpyWriter
from tqdm import tqdm

#拆分数据集
def split_records(input_record_filenames, output_train_file, output_eval_file, compression_type='GZIP'):
    print('split_records record...')
    options = RECORD.TFRecordOptions(compression_type=compression_type)
    dataset_reader = Loader.RandomDataset(input_record_filenames, options=options, with_share_memory=True).parse_from_numpy_writer()
    data_size = len(dataset_reader)
    all_example = []
    for i in tqdm(range(data_size), desc='load records'):
        serialized = dataset_reader[i]
        all_example.append(serialized)

    if hasattr(dataset_reader,'close'):
        dataset_reader.close()
    else:
        dataset_reader.reset()

    shuffle_idx = list(range(data_size))
    writer_train = NumpyWriter(output_train_file)
    writer_eval = NumpyWriter(output_eval_file)


    num_train = 0
    num_eval = 0
    for i in tqdm(shuffle_idx, desc='shuffle record'):
        example = all_example[i]

        if (i + 1) % 8 == 0:
            num_eval += 1
            count = num_eval
            writer = writer_eval
        else:
            num_train += 1
            count = num_train
            writer = writer_train
        #添加键值
        example['id'] = np.asarray(count - 1,dtype=np.int32)

        writer.write(example)

    writer_train.close()
    writer_eval.close()
    print('num_train',num_train,'num_eval',num_eval)


if __name__ == '__main__':



    example_files = r'/home/tk/train/make_big_data/output/dataset_0-train.record'

    output_train_file = os.path.join('/home/tk/train/make_big_data/output','train.record')

    output_eval_file = os.path.join('/home/tk/train/make_big_data/output','eval.record')

    split_records(input_record_filenames=example_files,
                  output_train_file=output_train_file,
                  output_eval_file=output_eval_file)
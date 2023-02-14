# @Time    : 2023/1/22 16:22
# @Author  : tk
# @FileName: data_utils.py
# 数据集 https://github.com/ssbuild/poetry_tang

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
    'model_type': 'bert',
    # 预训练模型路径 , 从0训练，则置空
    'model_name_or_path': '/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'tokenizer_name': '/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'config_name': '/data/nlp/pre_models/torch/bert/bert-base-chinese/config.json',
    'convert_onnx': False, # 转换onnx模型
    'do_train': True, 
    # 过滤诗集 poetry_85w_part1.record ，与唐诗宋词重复
    'train_file':  [_ for _ in gfile.glob('/data/nlp/nlp_train_data/poetry/*.record') if 'poetry_85w_part1.record' not in _],
    'max_epochs': 3,
    'train_batch_size': 8,
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

data_conf = {
    'stride': 50,
    'special': {
        '五绝': '[unused1]',
        '七绝': '[unused2]',
        '五律': '[unused3]',
        '七律': '[unused4]',
        '诗': '[unused5]',
        '花间集': '[unused6]',
        '幽梦影': '[unused6]',
        '词': '[unused6]',
        '论语': '[unused7]',
        '孟学': '[unused7]',
        '楚辞': '[unused7]',
        '诗经': '[unused7]',
        '四书五经': '[unused7]',
        '曲': '[unused8]',
        '歌词': '[unused9]',
        '对联': '[unused10]',
        '骂人': '[unused11]',
        '姓名': '[unused12]',
        '词语': '[unused13]',
        '成语': '[unused14]',
        '歇后语': '[unused15]',
        '汉字': '[unused16]',
        "先秦": '[unused17]',
        "秦": '[unused17]',
        "汉": '[unused17]',
        "魏晋": '[unused18]',
        "魏晋末南北朝初": '[unused18]',
        "隋": '[unused18]',
        "隋末唐初": '[unused18]',
        "唐": '[unused18]',
        "唐末宋初": '[unused18]',
        "南北朝": '[unused19]',
        "宋": '[unused19]',
        "宋末元初": '[unused19]',
        "宋末金初": '[unused19]',
        "辽": '[unused19]',
        "金": '[unused19]',
        "金末元初": '[unused20]',
        "元": '[unused20]',
        "元末明初": '[unused20]',
        "明": '[unused21]',
        "明末清初": '[unused21]',
        "清": '[unused21]',
        "清末民国初": '[unused21]',
        "清末近现代初": '[unused21]',
        "民国末当代初": '[unused22]',
        "近现代": '[unused22]',
        "近现代末当代初": '[unused22]',
        "当代": '[unused22]',
        "伤感网名": '[unused23]',
        "英文网名": '[unused24]',
        "女生网名": '[unused25]',
        "情侣网名": '[unused26]',
        "男生网名": '[unused27]',
        "搞笑网名": '[unused28]',
    }
}


def is_format(paragraphs: typing.List[typing.AnyStr]):
    length = 0
    flag = True
    for idx, sentence in enumerate(paragraphs):
        n = 0
        for char in sentence:
            if is_chinese_char(ord(char)):
                n += 1
        if idx == 0:
            length = n
            continue
        if n != length:
            flag = False
            break
    return flag


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
       
        sub_list = data
        input_ids = []
        # 每1千首
        for idx, (type, title, paragraphs) in enumerate(sub_list):
            text = type + title + '<T>' + paragraphs
            o = tokenizer.encode_plus(text, max_length=max_seq_length,
                                      truncation=True, return_attention_mask=False, return_token_type_ids=False)
            if len(o['input_ids']) <= 3:
                continue
            input_ids += o['input_ids'][1:-1]
            if idx != len(sub_list) - 1:
                input_ids += [tokenizer.sep_token_id]

        stride = data_conf['stride']
        pos = 0
        ds = []
        while pos < len(input_ids):
            input_ids_ = [tokenizer.cls_token_id] + input_ids[pos: pos + max_seq_length - 2] + [tokenizer.sep_token_id]
            pos += stride
            if len(input_ids_) <= 5:
                continue

            input_ids_ = np.asarray(input_ids_, dtype=np.int32)
            seqlen = np.asarray(len(input_ids_), dtype=np.int32)

            pad_len = max_seq_length - seqlen
            if pad_len:
                pad_val = tokenizer.pad_token_id
                input_ids_ = np.pad(input_ids_, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            d = {
                'input_ids': input_ids_,
                'seqlen': seqlen
            }
            ds.append(d)
        if self.index < 3:
            print(ds[0])
        return ds

        # 读取文件

    def on_get_corpus(self, files: typing.List, mode: str):
        D = []

        def poetry_parser(x):
            x = str(x['node'].tolist(), encoding='utf-8')
            x = json.loads(x)
            return x

        # 单条数据
        # {'author': '徐铉', 'title': '春尽日游后湖赠刘起居', 'paragraphs': ['今朝湖上送春归，万顷澄波照白髭。', '笑折残花劝君酒，金丹成熟是何时。'], 'tones': ['平平平仄仄平平，仄仄平平仄仄平。', '仄仄平平仄平仄，平平平仄仄平平。']}

        for file in files:
            dataset = Loader.RandomDataset(file, options=RECORD.TFRecordOptions(
                compression_type='GZIP')).parse_from_numpy_writer()
            dataset = dataset.map(poetry_parser)

            COUNT_PER_GROUP = 1000
            basename = os.path.basename(file)

            if basename == 'xm.record':  # 短数据，增大样本分组
                COUNT_PER_GROUP = 10000

            sub = []
            special = data_conf['special']
            for i in range(len(dataset)):
                d = dataset[i]
                title = d.get('title', '')
                paragraphs = d['paragraphs']
                data_type: str = d['type']
                type = special.get(data_type, None)
                if type is None:
                    data_type = data_type.replace('宋词', '词').replace('南唐词', '词').replace('元曲', '曲')
                    data_type = data_type.replace('宋', '').replace('唐', '')
                    type = ''
                    if data_type.endswith('诗') and is_format(paragraphs):
                        if len(paragraphs) == 2:
                            if paragraphs[0].find('，'):
                                length = len(paragraphs[0].split('，')[0])
                                if length == 5:
                                    type = special['五绝']
                                elif length == 7:
                                    type = special['七绝']
                        elif len(paragraphs) == 4:
                            if paragraphs[0].find('，'):
                                length = len(paragraphs[0].split('，')[0])
                                if length == 5:
                                    type = special['五律']
                                elif length == 7:
                                    type = special['七律']

                paragraphs = '<n>'.join(paragraphs)
                if len(paragraphs) == 0:
                    continue
                # 每1千首为一组
                if len(sub) < COUNT_PER_GROUP:
                    sub.append((type, title, paragraphs))
                else:
                    D.append(copy.deepcopy(sub))
                    sub.clear()
            if sub:
                D.append(copy.deepcopy(sub))
        return D

    def collate_fn(self,batch):
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

        seqlens = o.pop('seqlen')
        max_len = torch.max(seqlens)
        o['input_ids'] = o['input_ids'][:, :max_len].long()
        token_type_ids = torch.zeros_like(o['input_ids'], dtype=torch.long)
        for idx, seqlen in enumerate(seqlens):
            seqlen = seqlen.squeeze(-1).numpy().tolist()
            s = np.random.randint(2, seqlen - 1, dtype=np.int32).tolist()
            token_type_ids[idx, s:seqlen - 1] = 1
        o['token_type_ids'] = token_type_ids
        o['labels'] = torch.clone(o['input_ids'])
        return o


if __name__ == '__main__':

    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
    model_args, training_args, data_args = parser.parse_dict(train_info_args)

    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer, config, label2id, id2label = dataHelper.load_tokenizer_and_config()

    # 缓存数据集
    # 检测是否存在 output/dataset_0-train.record ，不存在则制作数据集
    if data_args.do_train:
        dataHelper.make_dataset_with_args(data_args.train_file, shuffle=True,mode='train')
    if data_args.do_eval:
        dataHelper.make_dataset_with_args(data_args.eval_file, mode='eval')
    if data_args.do_test:
        dataHelper.make_dataset_with_args(data_args.test_file,mode='test')


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

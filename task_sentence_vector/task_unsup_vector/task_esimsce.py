# -*- coding: utf-8 -*-
# 参考实现: https://github.com/shuxinyin/SimCSE-Pytorch
# 这里直接随机选了小于2万条任务数据训练；

import json
import logging
import random
import typing

import jieba
import numpy as np
import torch
from deep_training.data_helper import DataHelper
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from deep_training.nlp.models.esimcse import TransformerForESimcse
from deep_training.utils.trainer import SimpleModelCheckpoint
from fastdatasets.torch_dataset import Dataset as torch_Dataset
from pytorch_lightning import Trainer
from scipy import stats
from sklearn.metrics.pairwise import paired_distances
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import HfArgumentParser, BertTokenizer

train_info_args = {
    'devices': 1,
    'data_backend': 'record',
    'model_type': 'bert',
    'model_name_or_path': '/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'tokenizer_name': '/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'config_name': '/data/nlp/pre_models/torch/bert/bert-base-chinese/config.json',
    'convert_onnx': False, # 转换onnx模型
    'do_train': True, 
    'do_eval': True,
    # 'train_file': ['/data/nlp/nlp_train_data/clue/afqmc_public/train.json'],
    # 'eval_file': ['/data/nlp/nlp_train_data/clue/afqmc_public/dev.json'],
    # 'test_file': ['/data/nlp/nlp_train_data/clue/afqmc_public/test.json'],
    # 'train_file': ['/data/nlp/nlp_train_data/senteval_cn/LCQMC/LCQMC.train.data'],
    # 'eval_file': ['/data/nlp/nlp_train_data/senteval_cn/LCQMC/LCQMC.valid.data'],
    # 'test_file': ['/data/nlp/nlp_train_data/senteval_cn/LCQMC/LCQMC.test.data'],
    # 'train_file': ['/data/nlp/nlp_train_data/senteval_cn/STS-B/STS-B.train.data'],
    # 'eval_file': ['/data/nlp/nlp_train_data/senteval_cn/STS-B/STS-B.valid.data'],
    # 'test_file': ['/data/nlp/nlp_train_data/senteval_cn/STS-B/STS-B.test.data'],
    'train_file': [ '/data/nlp/nlp_train_data/senteval_cn/BQ/BQ.train.data'],
    'eval_file': [ '/data/nlp/nlp_train_data/senteval_cn/BQ/BQ.valid.data'],
    'test_file': [ '/data/nlp/nlp_train_data/senteval_cn/BQ/BQ.test.data'],
    # 'train_file': ['/data/nlp/nlp_train_data/senteval_cn/ATEC/ATEC.train.data'],
    # 'eval_file': ['/data/nlp/nlp_train_data/senteval_cn/ATEC/ATEC.valid.data'],
    # 'test_file': ['/data/nlp/nlp_train_data/senteval_cn/ATEC/ATEC.test.data'],
    'max_epochs': 3,
    'optimizer': 'adamw',
    'learning_rate': 3e-5,
    'train_batch_size': 20,
    'eval_batch_size': 20,
    'test_batch_size': 2,
    'adam_epsilon': 1e-8,
    'gradient_accumulation_steps': 1,
    'max_grad_norm': 1.0,
    'weight_decay': 0,
    'warmup_steps': 0,
    'output_dir': './output',
    'train_max_seq_length': 80,
    'eval_max_seq_length': 80,
    'test_max_seq_length': 80,
}

# cls , pooler , last-avg , first-last-avg
pooling = 'cls'
data_cut_config = {
    'qb_size': 4,
    'dup_rate': 0.15
}


class DataCut(object):
    # qb_size 为缓存batch_size
    def __init__(self, tokenizer, qb_size=10, dup_rate=0.15):
        self.q = []
        self.qb_size = qb_size
        self.dup_rate = dup_rate
        self.tokenizer = tokenizer

    def word_repetition_normal(self, batch_text):
        dst_text = list()
        for text in batch_text:
            actual_len = len(text)
            dup_len = random.randint(a=0, b=max(
                2, int(self.dup_rate * actual_len)))
            dup_word_index = random.sample(
                list(range(1, actual_len)), k=min(dup_len, actual_len - 1))

            dup_text = ''
            for index, word in enumerate(text):
                dup_text += word
                if index in dup_word_index:
                    dup_text += word
            dst_text.append(dup_text)
        return dst_text

    def word_repetition_chinese(self, batch_text):
        ''' span duplicated for chinese
        '''
        dst_text = list()
        for text in batch_text:
            cut_text = jieba.cut(text, cut_all=False)
            text = list(cut_text)

            actual_len = len(text)
            dup_len = random.randint(a=0, b=max(
                2, int(self.dup_rate * actual_len)))
            dup_word_index = random.sample(
                list(range(1, actual_len)), k=min(dup_len, actual_len - 1))

            dup_text = ''
            for index, word in enumerate(text):
                dup_text += word
                if index in dup_word_index:
                    dup_text += word
            dst_text.append(dup_text)
        return dst_text

    def cache_negative_samples(self, batch) -> typing.Optional[typing.Dict]:
        negative_samples = None
        if len(self.q) > 0:
            negative_samples = self.q[:self.qb_size]
            # print("size of negative_samples", len(negative_samples))

        if len(self.q) + 1 >= self.qb_size:
            self.q.pop(0)
        self.q.append(batch)
        return negative_samples


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
        do_lower_case = tokenizer.do_lower_case
        label2id = self.label2id

        data_cut: DataCut = self.data_cut
        sentence1, sentence2, label_str = data
        if mode == 'train':
            ds = []
            sentence_poses = data_cut.word_repetition_normal([sentence1, sentence2])
            for sentence_item in [(sentence1, sentence_poses[0]), (sentence2, sentence_poses[1])]:
                o_list = []
                for sentence in sentence_item:
                    o = tokenizer(sentence, max_length=max_seq_length, truncation=True, add_special_tokens=True,
                                  return_token_type_ids=False)
                    for k in o:
                        o[k] = np.asarray(o[k], dtype=np.int32)
                    seqlen = np.asarray(len(o['input_ids']), dtype=np.int32)
                    o['seqlen'] = seqlen
                    pad_len = max_seq_length - seqlen
                    if pad_len > 0:
                        pad_val = tokenizer.pad_token_id
                        o['input_ids'] = np.pad(o['input_ids'], pad_width=(0, pad_len),
                                                constant_values=(pad_val, pad_val))
                        o['attention_mask'] = np.pad(o['attention_mask'], pad_width=(0, pad_len),
                                                     constant_values=(0, 0))
                    o_list.append(o)
                seqlen = np.max([o.pop('seqlen') for o in o_list])
                d = {k: np.stack([o_list[0][k], o_list[1][k]], axis=0) for k in o_list[0].keys()}
                d['seqlen'] = np.asarray(seqlen, dtype=np.int32)
                ds.append(d)
            return ds
        # 验证
        else:
            ds = {}
            for sentence in [sentence1, sentence2]:
                o = tokenizer(sentence, max_length=max_seq_length, truncation=True, add_special_tokens=True,
                              return_token_type_ids=False)
                for k in o:
                    o[k] = np.asarray(o[k], dtype=np.int32)
                seqlen = np.asarray(len(o['input_ids']), dtype=np.int32)
                pad_len = max_seq_length - seqlen
                if pad_len > 0:
                    pad_val = tokenizer.pad_token_id
                    o['input_ids'] = np.pad(o['input_ids'], pad_width=(0, pad_len), constant_values=(pad_val, pad_val))
                    o['attention_mask'] = np.pad(o['attention_mask'], pad_width=(0, pad_len), constant_values=(0, 0))
                d = {
                    'input_ids': o['input_ids'],
                    'attention_mask': o['attention_mask'],
                    'seqlen': seqlen
                }
                if 'input_ids' not in ds:
                    ds = d
                else:
                    for k, v in d.items():
                        ds[k + '2'] = v
            if label_str is not None:
                labels = np.asarray(int(label_str), dtype=np.int32)
                ds['labels'] = labels
            return ds

    # 读取标签
    def on_get_labels(self, files: typing.List[str]):
        return None, None

    # 读取文件
    def on_get_corpus(self, files: typing.List, mode: str):
        D = []
        for filename in files:
            with open(filename, mode='r', encoding='utf-8') as f:
                lines = f.readlines()
                if filename.endswith('.json'):
                    for line in lines:
                        jd = json.loads(line)
                        if not jd:
                            continue
                        D.append((jd['sentence1'], jd['sentence2'], jd.get('label', None)))
                else:
                    for line in lines:
                        line = line.replace('\r\n', '').replace('\n', '')
                        s1, s2, l = line.split('\t', 2)
                        D.append((s1, s2, l))
        # 训练数据重排序
        if mode == 'train':
            tmp = []
            for item in D:
                tmp.append(item[0])
                tmp.append(item[1])
            random.shuffle(tmp)
            D.clear()
            for item1, item2 in zip(tmp[::2], tmp[1::2]):
                D.append((item1, item2, None))
        return D


    def train_collate_fn(self,batch):
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
        max_len = torch.max(o.pop('seqlen'))
        o['input_ids'] = o['input_ids'][:, :, :max_len]
        o['attention_mask'] = o['attention_mask'][:, :, :max_len]
        # 产生负样本
        batch_neg_samples = self.external_kwargs['data_cut'].cache_negative_samples(
            {k: torch.clone(v[:, 0]) for k, v in o.items()})

        o['neg_num'] = torch.tensor(len(batch_neg_samples) if batch_neg_samples is not None else 0, dtype=torch.int32)
        if batch_neg_samples is not None:
            for i, samples in enumerate(batch_neg_samples):
                for k, v in samples.items():
                    o[k + str(i)] = v
        return o

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
        max_len = torch.max(o.pop('seqlen'))
        o['input_ids'] = o['input_ids'][:, :max_len]
        o['attention_mask'] = o['attention_mask'][:, :max_len]
        if 'seqlen2' in o:
            max_len = torch.max(o.pop('seqlen2'))
            o['input_ids2'] = o['input_ids2'][:, :max_len]
            o['attention_mask2'] = o['attention_mask2'][:, :max_len]
        return o


class MyTransformer(TransformerForESimcse, with_pl=True):
    def __init__(self, *args, **kwargs):
        super(MyTransformer, self).__init__(*args, **kwargs)
        self.pooling = pooling
        # config = self.config


def evaluate_sample(a_vecs, b_vecs, labels):
    print('*' * 30, 'evaluating...', a_vecs.shape, b_vecs.shape, labels.shape, 'pos', np.sum(labels))
    sims = 1 - paired_distances(a_vecs, b_vecs, metric='cosine')
    print(np.concatenate([sims[:5], sims[-5:]], axis=0))
    print(np.concatenate([labels[:5], labels[-5:]], axis=0))
    correlation, _ = stats.spearmanr(labels, sims)
    print('spearman ', correlation)
    return correlation


class MySimpleModelCheckpoint(SimpleModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super(MySimpleModelCheckpoint, self).__init__(*args, **kwargs)
        self.weight_file = './best.pt'

    def on_save_model(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        pl_module: MyTransformer

        # 当前设备
        device = torch.device('cuda:{}'.format(trainer.global_rank))
        eval_datasets = dataHelper.load_dataset(dataHelper.eval_files)
        eval_datasets = DataLoader(eval_datasets, batch_size=training_args.eval_batch_size,
                                   collate_fn=dataHelper.collate_fn)

        a_vecs, b_vecs, labels = [], [], []
        for i, batch in tqdm(enumerate(eval_datasets), total=len(eval_datasets), desc='evalute'):
            for k in batch:
                batch[k] = batch[k].to(device)
            o = pl_module.validation_step(batch, i)
            a_logits, b_logits, b_labels = o['outputs']
            for j in range(len(b_logits)):
                logit1 = np.asarray(a_logits[j], dtype=np.float32)
                logit2 = np.asarray(b_logits[j], dtype=np.float32)
                label = np.asarray(b_labels[j], dtype=np.int32)

                a_vecs.append(logit1)
                b_vecs.append(logit2)
                labels.append(label)

        a_vecs = np.stack(a_vecs, axis=0)
        b_vecs = np.stack(b_vecs, axis=0)
        labels = np.stack(labels, axis=0)

        corrcoef = evaluate_sample(a_vecs, b_vecs, labels)
        f1 = corrcoef
        best_f1 = self.best.get('f1', -np.inf)
        print('current', f1, 'best', best_f1)
        if f1 >= best_f1:
            self.best['f1'] = f1
            logging.info('save best {}, {}\n'.format(self.best['f1'], self.weight_file))
            trainer.save_checkpoint(self.weight_file)


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
    model_args, training_args, data_args = parser.parse_dict(train_info_args)

    checkpoint_callback = MySimpleModelCheckpoint(monitor="f1", every_n_epochs=1,
                                                  every_n_train_steps=300 // training_args.gradient_accumulation_steps)
    trainer = Trainer(
        log_every_n_steps=20,
        callbacks=[checkpoint_callback],
        max_epochs=training_args.max_epochs,
        max_steps=training_args.max_steps,
        accelerator="gpu",
        devices=data_args.devices,
        enable_progress_bar=True,
        default_root_dir=data_args.output_dir,
        gradient_clip_val=training_args.max_grad_norm,
        accumulate_grad_batches=training_args.gradient_accumulation_steps,
        num_sanity_val_steps=0,
        strategy='ddp' if torch.cuda.device_count() > 1 else None,
    )

    data_cut = DataCut(**data_cut_config)
    dataHelper = NN_DataHelper(data_args.data_backend,data_cut = data_cut)
    tokenizer, config, label2id, id2label = dataHelper.load_tokenizer_and_config(model_args, training_args, data_args)
    

    # 缓存数据集
    if data_args.do_train:
        dataHelper.make_dataset_with_args(data_args.train_file,
                                          data_args,shuffle=True,
                                          mode='train')
    if data_args.do_eval:
        dataHelper.make_dataset_with_args(data_args.eval_file,
                                          data_args,
                                          shuffle=False,
                                          mode='eval')
    if data_args.do_test:
        dataHelper.make_dataset_with_args(data_args.test_file,data_args, shuffle=False,mode='test')

    # 修改config的dropout系数
    config.attention_probs_dropout_prob = 0.3
    config.hidden_dropout_prob = 0.3
    model = MyTransformer(pooling=pooling, config=config, model_args=model_args, training_args=training_args)

    if not data_args.convert_onnx:
        train_datasets = dataHelper.load_dataset(dataHelper.train_files, shuffle=True, num_processes=trainer.world_size,
                                                 process_index=trainer.global_rank, infinite=True,
                                                 with_record_iterable_dataset=False,
                                                 with_load_memory=True, with_torchdataset=False)

        if train_datasets is not None:
            train_datasets = torch_Dataset(train_datasets.limit(20000))
            train_datasets = DataLoader(train_datasets, batch_size=training_args.train_batch_size,
                                        collate_fn=dataHelper.train_collate_fn,
                                        shuffle=False if isinstance(train_datasets, IterableDataset) else True)

        if train_datasets is not None:
            trainer.fit(model, train_dataloaders=train_datasets)
        else:
            eval_datasets = dataHelper.load_sequential_sampler(dataHelper.eval_files,batch_size=training_args.eval_batch_size,collate_fn=dataHelper.collate_fn)
            test_datasets = dataHelper.load_sequential_sampler(dataHelper.test_files,batch_size=training_args.test_batch_size,collate_fn=dataHelper.collate_fn)
            if eval_datasets is not None:
                trainer.validate(model, dataloaders=eval_datasets, ckpt_path='./best.pt')

            if test_datasets is not None:
                trainer.test(model, dataloaders=test_datasets, ckpt_path='best.pt')

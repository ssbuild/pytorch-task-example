# -*- coding: utf-8 -*-
import copy
import json
import logging
import os
import typing

import numpy as np
import torch
from deep_training.data_helper import DataHelper
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from deep_training.nlp.losses.focal_loss import FocalLoss
from deep_training.nlp.losses.loss_arcface import ArcMarginProduct
from deep_training.nlp.models.transformer import TransformerModel
from deep_training.utils.trainer import SimpleModelCheckpoint
from pytorch_lightning import Trainer
from scipy import stats
from sklearn.metrics.pairwise import paired_distances
from tfrecords import TFRecordOptions
from torch import nn
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import HfArgumentParser, BertTokenizer

train_info_args = {
    'devices': 1,
    'data_backend': 'memory_raw',
    'model_type': 'bert',
    'model_name_or_path': '/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'tokenizer_name': '/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'config_name': '/data/nlp/pre_models/torch/bert/bert-base-chinese/config.json',
    'do_train': True,
    'do_eval': True,
    'train_file': '/data/nlp/nlp_train_data/clue/tnews/train.json',
    'eval_file': '/data/nlp/nlp_train_data/clue/tnews/dev.json',
    'test_file': '/data/nlp/nlp_train_data/clue/tnews/test.json',
    'label_file': '/data/nlp/nlp_train_data/clue/tnews/labels.json',
    'learning_rate': 5e-5,
    'max_epochs': 30,
    'train_batch_size': 80,
    'eval_batch_size': 40,
    'test_batch_size': 40,
    'adam_epsilon': 1e-8,
    'gradient_accumulation_steps': 1,
    'max_grad_norm': 1.0,
    'weight_decay': 0,
    'warmup_steps': 0,
    'output_dir': './output',
    'max_seq_length': 128
}

# cls , pooler , last-avg , first-last-avg , reduce
pooling = 'cls'


class NN_DataHelper(DataHelper):
    # 切分词
    def on_data_process(self, data: typing.Any, mode: str):
        tokenizer: BertTokenizer
        max_seq_length = self.max_seq_length_dict[mode]
        tokenizer = self.tokenizer
        do_lower_case = tokenizer.do_lower_case
        label2id = self.label2id
        sentence, label_str = data

        o = tokenizer(sentence, max_length=max_seq_length, truncation=True, add_special_tokens=True, )
        input_ids = np.asarray(o['input_ids'], dtype=np.int64)
        attention_mask = np.asarray(o['attention_mask'], dtype=np.int64)

        labels = np.asarray(label2id[label_str] if label_str is not None else 0, dtype=np.int64)
        seqlen = np.asarray(len(input_ids), dtype=np.int64)
        pad_len = max_seq_length - len(input_ids)
        if pad_len > 0:
            pad_val = tokenizer.pad_token_id
            input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            attention_mask = np.pad(attention_mask, (0, pad_len), 'constant', constant_values=(0, 0))
        d = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': np.expand_dims(labels, 0),
            'seqlen': seqlen
        }
        return d

    # 读取标签
    def on_get_labels(self, files: typing.List[str]):
        if files is None:
            return None, None
        label_fname = files[0]
        is_json_file = label_fname.endswith('.json')
        D = []
        with open(label_fname, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace('\r\n', '').replace('\n', '')
                if not line: continue
                if is_json_file:
                    jd = json.loads(line)
                    line = jd['label']
                D.append(line)
        D = sorted(list(set(D)))
        label2id = {label: i for i, label in enumerate(D)}
        id2label = {i: label for i, label in enumerate(D)}
        return label2id, id2label

    # 读取文件
    def on_get_corpus(self, files: typing.List, mode: str):
        D = []
        for filename in files:
            with open(filename, mode='r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    jd = json.loads(line)
                    if not jd:
                        continue
                    D.append((jd['sentence'], jd.get('label', None)))
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
        max_len = torch.max(o.pop('seqlen'))
        o['input_ids'] = o['input_ids'][:, :max_len]
        o['attention_mask'] = o['attention_mask'][:, :max_len]
        if 'token_type_ids' in o:
            o['token_type_ids'] = o['token_type_ids'][:, :max_len]
        if 'seqlen2' in o:
            max_len = torch.max(o.pop('seqlen2'))
            o['input_ids2'] = o['input_ids2'][:, :max_len]
            o['attention_mask2'] = o['attention_mask2'][:, :max_len]
            if 'token_type_ids2' in o:
                o['token_type_ids2'] = o['token_type_ids2'][:, :max_len]
        return o


def generate_pair_example(all_example_dict: dict):
    all_example_dict = copy.copy(all_example_dict)

    all_example_pos, all_example_neg = [], []
    all_keys = list(all_example_dict.keys())
    np.random.shuffle(all_keys)

    num_all = 0
    for k, v in all_example_dict.items():
        num_all += len(v)
    pos_num_max = num_all // 2 // 5
    for pos_label in all_keys:
        examples = all_example_dict[pos_label]
        if len(examples) == 0:
            continue
        num_size = int(len(examples) // 5) if len(examples) > 100 else np.random.randint(1, min(50, len(examples)),
                                                                                         dtype=np.int32)
        if num_size < 2:
            continue
        id_list = list(range(len(examples)))
        ids = np.random.choice(id_list, replace=False, size=num_size)
        ids = sorted(ids, reverse=True)

        flag = False
        for i1, i2 in zip(ids[::2], ids[1::2]):
            v1 = examples[i1]
            v2 = examples[i2]
            examples.pop(i1)
            examples.pop(i2)
            all_example_pos.append((v1, v2))
            if len(all_example_pos) >= pos_num_max:
                break
        # 去除空标签数据
        if len(examples) <= 1:
            all_keys.remove(pos_label)
        if flag:
            break

    flat_examples = []
    for k in all_keys:
        d_list = all_example_dict[k]
        for d in d_list:
            flat_examples.append((k, d))
    print('construct neg from {} flat_examples'.format(len(flat_examples)))
    idx_list = list(range(len(flat_examples)))
    np.random.shuffle(idx_list)
    while len(idx_list) >= 2:
        flag = False
        k1, e1 = flat_examples[idx_list.pop(0)]
        for i in idx_list[1:]:
            k2, e2 = flat_examples[i]
            if k1 != k2:
                all_example_neg.append((e1, e2))
                idx_list.remove(i)
                if len(all_example_neg) > len(all_example_pos) * 5:
                    flag = True
                    break
                break
        if flag:
            break
    print('pos num', len(all_example_pos), 'neg num', len(all_example_neg))
    return all_example_pos, all_example_neg


def evaluate_sample(a_vecs, b_vecs, labels):
    print('*' * 30, 'evaluating...', a_vecs.shape, b_vecs.shape, labels.shape, 'pos', np.sum(labels))
    sims = 1 - paired_distances(a_vecs, b_vecs, metric='cosine')
    print(np.concatenate([sims[:5], sims[-5:]], axis=0))
    print(np.concatenate([labels[:5], labels[-5:]], axis=0))
    correlation, _ = stats.spearmanr(labels, sims)
    print('spearman ', correlation)
    return correlation


class MyTransformer(TransformerModel, with_pl=True):
    def __init__(self, *args, **kwargs):
        pooling = kwargs.pop('pooling', 'cls')
        super(MyTransformer, self).__init__(*args, **kwargs)
        self.pooling = pooling
        self.feat_head = nn.Linear(self.config.hidden_size, 512, bias=False)
        self.metric_product = ArcMarginProduct(512 if self.pooling == 'reduce' else 768, self.config.num_labels, s=30.0,
                                               m=0.50, easy_margin=False)

        loss_type = 'focal_loss'
        if loss_type == 'focal_loss':
            self.loss_fn = FocalLoss(gamma=2)
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss()

    def get_model_lr(self):
        return super(MyTransformer, self).get_model_lr() + [
            (self.feat_head, self.config.task_specific_params['learning_rate_for_task']),
            (self.metric_product, self.config.task_specific_params['learning_rate_for_task']),
            (self.loss_fn, self.config.task_specific_params['learning_rate_for_task'])
        ]

    def forward_for_hidden(self, *args, **batch):
        outputs = self.model(*args, **batch, output_hidden_states=True, )
        if self.pooling == 'cls':
            simcse_logits = outputs[0][:, 0]
        elif self.pooling == 'pooler':
            simcse_logits = outputs[1]
        elif self.pooling == 'last-avg':
            last = outputs[0].transpose(1, 2)  # [batch, 768, seqlen]
            simcse_logits = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        elif self.pooling == 'first-last-avg':
            first = outputs[2][1].transpose(1, 2)  # [batch, 768, seqlen]
            last = outputs[2][-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            simcse_logits = torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]
        elif self.pooling == 'reduce':
            simcse_logits = self.sim_head(outputs[1])
            simcse_logits = torch.tanh(simcse_logits)
        else:
            raise ValueError('not support pooling', self.pooling)
        return simcse_logits

    def compute_loss(self, *args, **batch) -> tuple:
        labels: torch.Tensor = batch.pop('labels', None)
        if self.model.training:
            logits = self.forward_for_hidden(*args, **batch)
            labels = torch.squeeze(labels, dim=1)
            metric_logits = self.metric_product(logits, labels)
            loss = self.loss_fn(metric_logits, labels)
            outputs = (loss.mean(), logits, labels)
        elif labels is not None:
            inputs = {}
            for k in list(batch.keys()):
                if k.endswith('2'):
                    inputs[k.replace('2', '')] = batch.pop(k)
            labels = torch.squeeze(labels, dim=1)
            logits = self.forward_for_hidden(*args, **batch)
            if inputs:
                logits2 = self.forward_for_hidden(*args, **inputs)
                outputs = (None, logits, logits2, labels)
            else:
                outputs = (None, logits, labels)
        else:
            logits = self.forward_for_hidden(*args, **batch)
            outputs = (logits,)
        return outputs


from fastdatasets.torch_dataset import Dataset as torch_Dataset
from fastdatasets import record


class MySimpleModelCheckpoint(SimpleModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super(MySimpleModelCheckpoint, self).__init__(*args, **kwargs)
        self.weight_file = './best.pt'

    def on_save_model(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        pl_module: MyTransformer
        options = TFRecordOptions(compression_type='GZIP')
        # 当前设备
        device = torch.device('cuda:{}'.format(trainer.global_rank))
        data_dir = os.path.dirname(data_args.eval_file[0])
        eval_pos_neg_cache_file = os.path.join(data_dir, 'eval_pos_neg.record.cache')
        # 生成缓存文件
        if not os.path.exists(eval_pos_neg_cache_file):
            eval_datasets = dataHelper.load_dataset(dataHelper.eval_files)
            all_data = [eval_datasets[i] for i in range(len(eval_datasets))]
            map_data = {}
            for d in all_data:
                label = np.squeeze(d['labels']).tolist()
                if label not in map_data:
                    map_data[label] = []
                map_data[label].append(d)
            pos_data, neg_data = generate_pair_example(map_data)
            # 生成缓存文件
            f_out = record.NumpyWriter(eval_pos_neg_cache_file, options=options)

            keep_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'seqlen']
            for pair in pos_data:
                for o in pair:
                    for k in list(o.keys()):
                        if k not in keep_keys:
                            o.pop(k)

                o = copy.copy(pair[0])
                for k, v in pair[1].items():
                    o[k + '2'] = v
                o['labels'] = np.asarray(1, dtype=np.int32)
                f_out.write(o)
            for pair in neg_data:
                for o in pair:
                    for k in list(o.keys()):
                        if k not in keep_keys:
                            o.pop(k)

                o = copy.copy(pair[0])
                for k, v in pair[1].items():
                    o[k + '2'] = v
                o['labels'] = np.asarray(0, dtype=np.int32)
                f_out.write(o)
            f_out.close()

        assert os.path.exists(eval_pos_neg_cache_file)
        eval_datasets_pos_neg = record.load_dataset.RandomDataset(eval_pos_neg_cache_file,
                                                                  options=options).parse_from_numpy_writer()
        eval_datasets = DataLoader(torch_Dataset(eval_datasets_pos_neg), batch_size=training_args.eval_batch_size,
                                   collate_fn=dataHelper.collate_fn)
        a_vecs, b_vecs, labels = [], [], []
        for i, batch in tqdm(enumerate(eval_datasets),
                             total=len(eval_datasets_pos_neg) // training_args.eval_batch_size, desc='evalute'):
            for k in batch:
                batch[k] = batch[k].to(device)
            o = pl_module.validation_step(batch, i)
            a_logits, b_logits, label = o['outputs']
            for j in range(len(b_logits)):
                a_vecs.append(np.asarray(a_logits[j], dtype=np.float32))
                b_vecs.append(np.asarray(b_logits[j], dtype=np.float32))
                labels.append(np.squeeze(label[j]) if np.ndim(label[j]) > 0 else label[j])

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

    checkpoint_callback = MySimpleModelCheckpoint(monitor="f1", every_n_epochs=1)
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
    dataHelper = NN_DataHelper(data_args.data_backend)
    tokenizer, config, label2id, id2label = dataHelper.load_tokenizer_and_config(model_args, training_args,
                                                                                 data_args)

    # 缓存数据集
    if data_args.do_train:
        dataHelper.make_dataset_with_args(data_args.train_file,
                                          data_args, shuffle=True,
                                          mode='train')
    if data_args.do_eval:
        dataHelper.make_dataset_with_args(data_args.eval_file,
                                          data_args,shuffle=False,
                                          mode='eval')
    if data_args.do_test:
        dataHelper.make_dataset_with_args(data_args.test_file,data_args,shuffle=False,mode='test')

    train_datasets = dataHelper.load_dataset(dataHelper.train_files, shuffle=True, num_processes=trainer.world_size,
                                             process_index=trainer.global_rank, infinite=True,
                                             with_record_iterable_dataset=True)

    if train_datasets is not None:
        train_datasets = DataLoader(train_datasets, batch_size=training_args.train_batch_size,
                                    collate_fn=dataHelper.collate_fn,
                                    shuffle=False if isinstance(train_datasets, IterableDataset) else True)

    model = MyTransformer(pooling=pooling, config=config, model_args=model_args, training_args=training_args)

    if train_datasets is not None:
        trainer.fit(model, train_dataloaders=train_datasets)
    else:
        eval_datasets = dataHelper.load_dataset(dataHelper.eval_files)
        test_datasets = dataHelper.load_dataset(dataHelper.test_files)
        if eval_datasets is not None:
            eval_datasets = DataLoader(eval_datasets, batch_size=training_args.eval_batch_size,
                                       collate_fn=dataHelper.collate_fn)
        if test_datasets is not None:
            test_datasets = DataLoader(test_datasets, batch_size=training_args.test_batch_size,
                                       collate_fn=dataHelper.collate_fn)
        if eval_datasets is not None:
            trainer.validate(model, dataloaders=eval_datasets, ckpt_path='./best.pt')

        if test_datasets is not None:
            trainer.test(model, dataloaders=test_datasets, ckpt_path='best.pt')

# -*- coding: utf-8 -*-
import copy
import json
import logging
import os.path
import typing

import numpy as np
import pytorch_lightning
import torch
from deep_training.data_helper import DataHelper
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from deep_training.data_helper import load_tokenizer_and_config_with_args
from deep_training.nlp.models.infonce import TransformerForInfoNce
from deep_training.utils.trainer import SimpleModelCheckpoint
from pytorch_lightning import Trainer
from scipy import stats
from sklearn.metrics.pairwise import paired_distances
from tfrecords import TFRecordOptions
from torch import nn
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import HfArgumentParser, BertTokenizer

# model_base_dir = '/data/torch/bert-base-chinese'
model_base_dir = '/data/nlp/pre_models/torch/bert/bert-base-chinese'

train_info_args = {
    'devices': torch.cuda.device_count(),
    'data_backend': 'record',
    'model_type': 'bert',
    'model_name_or_path': model_base_dir,
    'tokenizer_name': model_base_dir,
    'config_name': os.path.join(model_base_dir, 'config.json'),
    'do_train': True,
    'do_eval': True,
    'do_test': False,
     # 'train_file':'/data/nlp/nlp_train_data/senteval_cn/LCQMC/LCQMC.train.data',
    # 'eval_file':'/data/nlp/nlp_train_data/senteval_cn/LCQMC/LCQMC.valid.data',
    # 'test_file':'/data/nlp/nlp_train_data/senteval_cn/LCQMC/LCQMC.test.data',
    # 'train_file':'/data/nlp/nlp_train_data/senteval_cn/STS-B/STS-B.train.data',
    # 'eval_file':'/data/nlp/nlp_train_data/senteval_cn/STS-B/STS-B.valid.data',
    # 'test_file':'/data/nlp/nlp_train_data/senteval_cn/STS-B/STS-B.test.data',
    'train_file':'/data/nlp/nlp_train_data/senteval_cn/BQ/BQ.train.data',
    'eval_file':'/data/nlp/nlp_train_data/senteval_cn/BQ/BQ.valid.data',
    'test_file':'/data/nlp/nlp_train_data/senteval_cn/BQ/BQ.test.data',
    # 'train_file':'/data/nlp/nlp_train_data/senteval_cn/ATEC/ATEC.train.data',
    # 'eval_file':'/data/nlp/nlp_train_data/senteval_cn/ATEC/ATEC.valid.data',
    # 'test_file':'/data/nlp/nlp_train_data/senteval_cn/ATEC/ATEC.test.data',
    'learning_rate': 3e-5,
    'max_steps': 120000,
    'max_epochs': 1,
    'train_batch_size': 2,
    'eval_batch_size': 10,
    'test_batch_size': 1,
    'adam_epsilon': 1e-8,
    'gradient_accumulation_steps': 20,
    'max_grad_norm': 1.0,
    'weight_decay': 0,
    'warmup_steps': 0,
    'output_dir': './output',
    'train_max_seq_length': 512,
    'eval_max_seq_length': 512,
    'test_max_seq_length': 512,
}

#cls , pooler , last-avg , first-last-avg , reduce
pooling = 'cls'
temperature= 0.1


class NN_DataHelper(DataHelper):
    index = 1

    def on_data_ready(self):
        self.index = -1

    # 切分词
    def on_data_process(self, data: typing.Any, user_data: tuple):
        self.index += 1
        tokenizer: BertTokenizer
        tokenizer, max_seq_length, do_lower_case, label2id, mode = user_data
        # 训练集(sentence1, sentence2, sentence3)  验证集(sentence1, sentence2, labelstr)
        sentence1, sentence2, sentence3_or_labelstr = data
        if mode == 'train':
            o_list = []
            for sentence in [sentence1, sentence2, sentence3_or_labelstr]:
                if sentence is None:  # 无负样本
                    continue
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
                o_list.append(copy.deepcopy(d))

            seqlen = np.max([o.pop('seqlen') for o in o_list])
            d = {k: np.stack([o_list[0][k], o_list[1][k]], axis=0) for k in o_list[0].keys()}
            d['seqlen'] = np.asarray(seqlen, dtype=np.int32)
            return d
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
            if sentence3_or_labelstr is not None:
                labels = np.asarray(int(sentence3_or_labelstr), dtype=np.int32)
                ds['labels'] = labels
            return ds

    # 读取标签
    def on_get_labels(self, files: typing.List[str]):
        return None,None

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
                        if mode == 'train':
                            if 'sentence3' in jd:
                                D.append((jd['sentence1'], jd['sentence2'], jd['sentence3']))
                            else:
                                D.append((jd['sentence1'], jd['sentence2'], None))
                        else:
                            D.append((jd['sentence1'], jd['sentence2'], jd['label']))
                else:
                    for line in lines:
                        line = line.replace('\r\n', '').replace('\n', '')
                        s1, s2, s3 = line.split('\t', 2)
                        if mode == 'train':
                            s3: str
                            if s3.isdigit() or s3.isdecimal() or s3.isnumeric():
                                D.append((s1, s2, None))
                            else:
                                D.append((s1, s2, s3))
                        else:
                            D.append((s1, s2, 1))
                            D.append((s1, s3, 0))

        return D

    @staticmethod
    def collate_fn(batch):
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
        num_size = int(len(examples) // 5) if len(examples) > 100 else np.random.randint(1,min(50,len(examples)),dtype=np.int32)
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


def evaluate_sample(a_vecs,b_vecs,labels):
    print('*' * 30,'evaluating....',a_vecs.shape,b_vecs.shape,labels.shape)
    sims = 1 - paired_distances(a_vecs,b_vecs,metric='cosine')
    print(np.concatenate([sims[:5] , sims[-5:]],axis=0))
    print(np.concatenate([labels[:5] , labels[-5:]],axis=0))
    correlation,_  = stats.spearmanr(labels,sims)
    print('spearman ', correlation)
    return correlation

class MyTransformer(TransformerForInfoNce, pytorch_lightning.LightningModule, with_pl=True):
    def __init__(self,*args, **kwargs):
        super(MyTransformer, self).__init__(*args, **kwargs)




from fastdatasets.torch_dataset import Dataset as torch_Dataset
from fastdatasets import record
class MySimpleModelCheckpoint(SimpleModelCheckpoint):
    def __init__(self,*args,**kwargs):
        super(MySimpleModelCheckpoint, self).__init__(*args,**kwargs)
        self.weight_file = './best.pt'

    def on_save_model(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        pl_module: MyTransformer
        options = TFRecordOptions(compression_type='GZIP')
        #当前设备
        device = torch.device('cuda:{}'.format(trainer.global_rank))
        data_dir = os.path.dirname(data_args.eval_file[0])
        eval_pos_neg_cache_file = os.path.join(data_dir, 'eval_pos_neg.record.cache')
        # 缓存文件
        if os.path.exists(eval_pos_neg_cache_file):
            eval_datasets_pos_neg = record.load_dataset.RandomDataset(eval_pos_neg_cache_file,
                                                                      options=options).parse_from_numpy_writer()
            pos_data, neg_data = [], []
            for o in eval_datasets_pos_neg:
                obj_list = pos_data if np.squeeze(o.pop('positive')) > 0 else neg_data
                keys1, keys2 = [k for k in o if not k.endswith('2')], [k for k in o if k.endswith('2')]
                d1 = {k: o[k] for k in keys1}
                d2 = {k.replace('2', ''): o[k] for k in keys2}
                obj_list.append((d1,d2))
            print('pos num', len(pos_data) , 'neg num', len(neg_data) )
        else:
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
            for pair in pos_data:
                o = copy.copy(pair[0])
                for k, v in pair[1].items():
                    o[k + '2'] = v
                o['positive'] = np.asarray(1, dtype=np.int32)
                f_out.write(o)
            for pair in neg_data:
                o = copy.copy(pair[0])
                for k, v in pair[1].items():
                    o[k + '2'] = v
                o['positive'] = np.asarray(0, dtype=np.int32)
                f_out.write(o)
            f_out.close()

        a_data = [_[0] for _ in pos_data + neg_data]
        b_data = [_[1] for _ in pos_data + neg_data]
        labels = np.concatenate([np.ones(len(pos_data),dtype=np.int32),np.zeros(len(neg_data),dtype=np.int32)])
        t_data = a_data + b_data
        eval_datasets = DataLoader(torch_Dataset(t_data), batch_size=training_args.eval_batch_size,collate_fn=dataHelper.collate_fn)
        vecs = []
        for i,batch in tqdm(enumerate(eval_datasets),total=len(t_data)//training_args.eval_batch_size,desc='evalute'):
            for k in batch:
                batch[k] = batch[k].to(device)
            o = pl_module.validation_step(batch,i)
            b_logits, _ = o['outputs']
            for j in range(len(b_logits)):
                logit = np.asarray(b_logits[j], dtype=np.float32)
                vecs.append(logit)

        a_vecs = np.stack(vecs[:len(a_data)],axis=0)
        b_vecs = np.stack(vecs[len(a_data):],axis=0)

        corrcoef = evaluate_sample(a_vecs,b_vecs,labels)
        f1 = corrcoef
        best_f1 = self.best.get('f1',-np.inf)
        print('current', f1, 'best', best_f1)
        if f1 >= best_f1:
            self.best['f1'] = f1
            logging.info('save best {}, {}\n'.format(self.best['f1'], self.weight_file))
            trainer.save_checkpoint(self.weight_file)



if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
    model_args, training_args, data_args = parser.parse_dict(train_info_args)

    checkpoint_callback = MySimpleModelCheckpoint(every_n_epochs=1,
                                                  every_n_train_steps=500 // training_args.gradient_accumulation_steps)
    trainer = Trainer(
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
    tokenizer, config, label2id, id2label = load_tokenizer_and_config_with_args(dataHelper, model_args, training_args,
                                                                                data_args)

    token_fn_args_dict = {
        'train': (tokenizer, data_args.train_max_seq_length, model_args.do_lower_case, label2id, 'train'),
        'eval': (tokenizer, data_args.eval_max_seq_length, model_args.do_lower_case, label2id, 'eval'),
        'test': (tokenizer, data_args.test_max_seq_length, model_args.do_lower_case, label2id, 'test')
    }

    # 缓存数据集
    intermediate_name = data_args.intermediate_name + '_{}'.format(0)
    if data_args.do_train:
        dataHelper.train_files.append(
            dataHelper.make_dataset_with_args(data_args.train_file, token_fn_args_dict['train'],
                                              data_args,
                                              intermediate_name=intermediate_name, shuffle=True,
                                              mode='train'))
    if data_args.do_eval:
        dataHelper.eval_files.append(dataHelper.make_dataset_with_args(data_args.eval_file, token_fn_args_dict['eval'],
                                                                       data_args,
                                                                       intermediate_name=intermediate_name,
                                                                       shuffle=False,
                                                                       mode='eval'))
    if data_args.do_test:
        dataHelper.test_files.append(dataHelper.make_dataset_with_args(data_args.test_file, token_fn_args_dict['test'],
                                                                       data_args,
                                                                       intermediate_name=intermediate_name,
                                                                       shuffle=False,
                                                                       mode='test'))

    train_datasets = dataHelper.load_dataset(dataHelper.train_files, shuffle=True, num_processes=trainer.world_size,
                                             process_index=trainer.global_rank, infinite=True,
                                             with_record_iterable_dataset=False,
                                             with_load_memory=True, with_torchdataset=True)

    if train_datasets is not None:
        train_datasets = DataLoader(train_datasets, batch_size=training_args.train_batch_size,
                                    collate_fn=dataHelper.collate_fn,
                                    shuffle=False if isinstance(train_datasets, IterableDataset) else True)

    model = MyTransformer(pooling=pooling,temperature=temperature,config=config, model_args=model_args, training_args=training_args)

    if train_datasets is not None:
        trainer.fit(model,train_dataloaders=train_datasets)

    else:
        #加载权重
        model = MyTransformer.load_from_checkpoint('./best.pt', pooling=pooling, temperature=temperature,
                                                   config=config, model_args=model_args,
                                                   training_args=training_args)

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
            trainer.test(model, dataloaders=test_datasets, ckpt_path='./best.pt')


        is_convert_onnx = True
        #是否转换模型
        if is_convert_onnx:
            input_sample = (
                torch.ones(size=(1, 128), dtype=torch.int32),
                torch.ones(size=(1, 128), dtype=torch.int32),
            )
            model.eval()
            model.to('cuda')
            input_names = ["input_ids", "attention_mask"]
            out_names = ["pred_ids"]

            model = MyTransformer.load_from_checkpoint('./best.pt',pooling=pooling,temperature=temperature,
                                                       config=config, model_args=model_args,
                                                       training_args=training_args)
            model.to_onnx('./best.onnx',
                          input_sample=input_sample,
                          verbose=True,
                          opset_version=10,
                          do_constant_folding=True,
                          input_names=input_names,
                          output_names=out_names,
                          dynamic_axes={"input_ids": [0, 1],
                                        "attention_mask": [0, 1],
                                        "pred_ids": [0, 1]
                                        }
                          )


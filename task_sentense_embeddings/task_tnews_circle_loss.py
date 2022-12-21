# -*- coding: utf-8 -*-
import json
import typing

import numpy as np
import scipy
import torch
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch.nn import functional as F
from deep_training.data_helper import DataHelper
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from deep_training.data_helper import load_tokenizer_and_config_with_args
from deep_training.nlp.losses.circle_loss import CircleLoss
from deep_training.nlp.models.transformer import TransformerModel, TransformerMeta
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import HfArgumentParser, BertTokenizer

train_info_args = {
    'devices':  1,
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
    'train_batch_size': 64,
    'test_batch_size': 32,
    'adam_epsilon': 1e-8,
    'gradient_accumulation_steps': 10,
    'max_grad_norm': 1.0,
    'weight_decay': 0,
    'warmup_steps': 0,
    'output_dir': './output',
    'max_seq_length': 128
}

class NN_DataHelper(DataHelper):
    # 切分词
    def on_data_process(self,data: typing.Any, user_data: tuple):
        tokenizer: BertTokenizer
        tokenizer, max_seq_length, do_lower_case, label2id, mode = user_data
        sentence,label_str = data

        o = tokenizer(sentence, max_length=max_seq_length, truncation=True, add_special_tokens=True, )
        input_ids = np.asarray(o['input_ids'], dtype=np.int64)
        attention_mask = np.asarray(o['attention_mask'], dtype=np.int64)

        labels = np.asarray(label2id[label_str] if label_str is not None else 0,dtype=np.int64)
        seqlen = np.asarray(len(input_ids), dtype=np.int64)
        pad_len = max_seq_length - len(input_ids)
        if pad_len > 0:
            pad_val = tokenizer.pad_token_id
            input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            attention_mask = np.pad(attention_mask, (0, pad_len), 'constant', constant_values=(0, 0))
        d = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': np.expand_dims(labels,0),
            'seqlen': seqlen
        }
        return d

    #读取标签
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
    def on_get_corpus(self, files: typing.List, mode:str):
        D = []
        for filename in files:
            with open(filename, mode='r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    jd = json.loads(line)
                    if not jd:
                        continue
                    D.append((jd['sentence'], jd.get('label',None)))
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
        if 'token_type_ids' in o:
            o['token_type_ids'] = o['token_type_ids'][:, :max_len]
        return o

def transform_and_normalize(vecs, kernel=None, bias=None):
    """应用变换，然后标准化
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
    return vecs / np.clip(norms, 1e-8, np.inf)

def compute_corrcoef(x, y):
    """Spearman相关系数
    """
    return scipy.stats.spearmanr(x, y).correlation

def choise_samples_from_classvectors(vec_maps : dict):
    a_vecs, b_vecs, labels = [], [], []
    for k in vec_maps:
        print(k, len(vec_maps[k]))
        obj_list = vec_maps[k]
        val = [obj_list[ids]
               for ids in
               np.random.choice(list(range(len(obj_list))), min(1000, len(obj_list)))
               ]
        if len(val) > 2:
            for j in range(0, len(val) // 2, 2):
                a_vecs.append(val[j])
                b_vecs.append(val[j + 1])
                labels.append(1)

    for k1 in vec_maps.keys():
        for k2 in vec_maps.keys():
            if k1 == k2:
                continue

            obj_list1 = vec_maps[k1]
            val1 = [obj_list1[ids]
                    for ids in
                    np.random.choice(list(range(len(obj_list1))), min(10, len(obj_list1)))
                    ]

            obj_list2 = vec_maps[k2]
            val2 = [obj_list2[ids]
                    for ids in
                    np.random.choice(list(range(len(obj_list2))), min(10, len(obj_list2)))
                    ]

            if val1 and val2:
                for j in range(min(len(val1), len(val2))):
                    a_vecs.append(val1[j])
                    b_vecs.append(val2[j])
                    labels.append(0)

    print('total sample', len(labels), 'pos', np.sum(labels))

    a_vecs = np.stack(a_vecs, axis=0)
    b_vecs = np.stack(b_vecs, axis=0)
    labels = np.stack(labels, axis=0)
    return a_vecs,b_vecs,labels

class MyTransformer(TransformerModel, metaclass=TransformerMeta):
    def __init__(self,*args,**kwargs):
        super(MyTransformer, self).__init__(*args,**kwargs)
        self.feat_head = nn.Linear(config.hidden_size, 512, bias=False)
        self.loss_fn = CircleLoss(m=0.25, gamma=64)


    def get_model_lr(self):
        return super(MyTransformer, self).get_model_lr() + [
            (self.feat_head, self.config.task_specific_params['learning_rate_for_task']),
            (self.loss_fn, self.config.task_specific_params['learning_rate_for_task'])
        ]

    def compute_loss(self,batch,batch_idx) -> tuple:
        labels: torch.Tensor = batch.pop('labels',None)
        outputs = self(**batch)
        logits = self.feat_head(outputs[0][:, 0, :])
        # logits = torch.tan(logits)
        # logits = F.normalize(logits)
        if labels is not None:
            labels = torch.squeeze(labels, dim=1)
            loss = self.loss_fn(F.normalize(logits),labels)
            outputs = (loss,logits,labels)
        else:
            outputs = (logits,)
        return outputs

    def validation_epoch_end(self, outputs: typing.Union[EPOCH_OUTPUT, typing.List[EPOCH_OUTPUT]]) -> None:
        print('test_epoch_end...')
        # from fastdatasets.record import NumpyWriter
        # f = NumpyWriter('./eval_vecs.record')
        # for i, o in tqdm(enumerate(outputs), total=len(outputs)):
        #     _,b_logits, b_labels = o['outputs']
        #     for j in range(len(b_logits)):
        #         obj =  {
        #             'logit': np.asarray(b_logits[j],dtype=np.float32),
        #             'label': np.asarray(b_labels[j],dtype=np.int32),
        #         }
        #         f.write(obj)
        # f.close()
        vec_maps = {}
        for i, o in tqdm(enumerate(outputs), total=len(outputs)):
            b_logits, b_labels = o['outputs']
            for j in range(len(b_logits)):
                logit = np.asarray(b_logits[j], dtype=np.float32)
                label = np.asarray(b_labels[j], dtype=np.int32)

                label = label.squeeze().tolist()
                if label not in vec_maps:
                    vec_maps[label] = []
                vec_maps[label].append(logit)
        eval_samples = choise_samples_from_classvectors(vec_maps)

        a_vecs, b_vecs, labels = eval_samples
        a_vecs = transform_and_normalize(a_vecs)
        b_vecs = transform_and_normalize(b_vecs)
        sims = (a_vecs * b_vecs).sum(axis=1)
        corrcoef = compute_corrcoef(labels, sims)

        print(corrcoef)
        self.log('corrcoef', corrcoef, prog_bar=True)

def get_trainer():
    checkpoint_callback = ModelCheckpoint(monitor="corrcoef", save_last=False, every_n_epochs=1)
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
    )
    return trainer

if __name__== '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
    model_args, training_args, data_args = parser.parse_dict(train_info_args)

    trainer = get_trainer()
    dataHelper = NN_DataHelper(data_args.data_backend)
    tokenizer, config, label2id, id2label = load_tokenizer_and_config_with_args(dataHelper, model_args, training_args,data_args)

    token_fn_args_dict = {
        'train': (tokenizer, data_args.train_max_seq_length, model_args.do_lower_case, label2id, 'train'),
        'eval': (tokenizer, data_args.eval_max_seq_length, model_args.do_lower_case, label2id, 'eval'),
        'test': (tokenizer, data_args.test_max_seq_length, model_args.do_lower_case, label2id, 'test')
    }

    N = 1
    train_files, eval_files, test_files = [], [], []
    for i in range(N):
        intermediate_name = data_args.intermediate_name + '_{}'.format(i)
        if data_args.do_train:
            train_files.append(
                dataHelper.make_dataset_with_args(data_args.train_file, token_fn_args_dict['train'], data_args,
                                       intermediate_name=intermediate_name, shuffle=True, mode='train'))
        if data_args.do_eval:
            eval_files.append(
                dataHelper.make_dataset_with_args(data_args.eval_file, token_fn_args_dict['eval'], data_args,
                                       intermediate_name=intermediate_name, shuffle=False, mode='eval'))
        if data_args.do_test:
            test_files.append(
                dataHelper.make_dataset_with_args(data_args.test_file, token_fn_args_dict['test'], data_args,
                                       intermediate_name=intermediate_name, shuffle=False, mode='test'))

    train_datasets = dataHelper.load_dataset(train_files,shuffle=True,num_processes=trainer.world_size,process_index=trainer.global_rank,infinite=True,with_record_iterable_dataset=True)
    eval_datasets = dataHelper.load_dataset(eval_files,num_processes=trainer.world_size,process_index=trainer.global_rank)
    test_datasets = dataHelper.load_dataset(test_files,num_processes=trainer.world_size,process_index=trainer.global_rank)
    if train_datasets is not None:
        train_datasets = DataLoader(train_datasets,batch_size=training_args.train_batch_size,collate_fn=dataHelper.collate_fn,shuffle=False if isinstance(train_datasets, IterableDataset) else True)
    if eval_datasets is not None:
        eval_datasets = DataLoader(eval_datasets,batch_size=training_args.eval_batch_size,collate_fn=dataHelper.collate_fn)
    if test_datasets is not None:
        test_datasets = DataLoader(test_datasets,batch_size=training_args.test_batch_size,collate_fn=dataHelper.collate_fn)

    

    model = MyTransformer(config=config,model_args=model_args,training_args=training_args)

    if train_datasets is not None:
        trainer.fit(model, train_dataloaders=train_datasets)

    if eval_datasets is not None:
        trainer.validate(model, dataloaders=eval_datasets)

    if test_datasets is not None:
        trainer.test(model, dataloaders=test_datasets)

# -*- coding: utf-8 -*-
import json
import logging
import typing

import numpy as np
import scipy
import torch
from deep_training.data_helper import DataHelper
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from deep_training.data_helper import load_tokenizer_and_config_with_args
from deep_training.nlp.losses.ContrastiveLoss import ContrastiveLoss
from deep_training.nlp.models.transformer import TransformerModel
from deep_training.utils.func import seq_pading
from deep_training.utils.trainer import SimpleModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from scipy import stats
from sklearn.metrics.pairwise import paired_distances
from torch import nn
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import HfArgumentParser, BertTokenizer

train_info_args = {
'devices':  1,
'data_backend':'memory_raw',
'model_type': 'bert',
'model_name_or_path':'/data/nlp/pre_models/torch/bert/bert-base-chinese',
'tokenizer_name':'/data/nlp/pre_models/torch/bert/bert-base-chinese',
'config_name':'/data/nlp/pre_models/torch/bert/bert-base-chinese/config.json',
'do_train': True,
'train_file':'/data/nlp/nlp_train_data/clue/afqmc_public/train.json',
'eval_file':'/data/nlp/nlp_train_data/clue/afqmc_public/dev.json',
'test_file':'/data/nlp/nlp_train_data/clue/afqmc_public/test.json',
'optimizer': 'adamw',
'learning_rate':5e-5,
'max_epochs':3,
'train_batch_size':64,
'test_batch_size':2,
'adam_epsilon':1e-8,
'gradient_accumulation_steps':1,
'max_grad_norm':1.0,
'weight_decay':0,
'warmup_steps':0,
'output_dir':'./output',
'max_seq_length':140
}


def pad_to_seqlength(sentence,tokenizer,max_seq_length):
    tokenizer: BertTokenizer
    o = tokenizer(sentence, max_length=max_seq_length, truncation=True, add_special_tokens=True, )
    arrs = [o['input_ids'],o['attention_mask']]
    seqlen = np.asarray(len(arrs[0]),dtype=np.int64)
    input_ids,attention_mask = seq_pading(arrs,max_seq_length=max_seq_length,pad_val=tokenizer.pad_token_id)
    return input_ids,attention_mask,seqlen

class NN_DataHelper(DataHelper):
    # 切分词
    def on_data_process(self,data: typing.Any, user_data: tuple):
        tokenizer: BertTokenizer
        tokenizer, max_seq_length, do_lower_case, label2id, mode = user_data

        sentence1,sentence2,label_str = data
        labels = np.asarray(1 - label2id[label_str] if label_str is not None else 0, dtype=np.int64)

        input_ids, attention_mask,seqlen = pad_to_seqlength(sentence1,tokenizer,max_seq_length)
        input_ids_2, attention_mask_2, seqlen_2 = pad_to_seqlength(sentence2, tokenizer, max_seq_length)
        d = {
            'labels': labels,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'seqlen': seqlen,
            'input_ids_2': input_ids_2,
            'attention_mask_2': attention_mask_2,
            'seqlen_2': seqlen_2
        }
        return d

    #读取标签
    def on_get_labels(self, files: typing.List[str]):
        D = ['0','1']
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
                    D.append((jd['sentence1'],jd['sentence2'], jd.get('label',None)))
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

        seqlen = o.pop('seqlen_2')
        max_len = torch.max(seqlen)
        o['input_ids_2'] = o['input_ids_2'][:, :max_len]
        o['attention_mask_2'] = o['attention_mask_2'][:, :max_len]

        return o

class MyTransformer(TransformerModel, with_pl=True):
    def __init__(self,*args,**kwargs):
        super(MyTransformer, self).__init__(*args,**kwargs)
        config = self.config
        self.feat_head = nn.Linear(config.hidden_size, 512, bias=False)
        self.loss_fn = ContrastiveLoss(size_average=False,margin=0.5)

    def get_model_lr(self):
        return super(MyTransformer, self).get_model_lr() + [
            (self.feat_head, self.config.task_specific_params['learning_rate_for_task'])
        ]

    def compute_loss(self, *args,**batch) -> tuple:
        labels: torch.Tensor = batch.pop('labels',None)
        if labels is not None:
            batch2 = {
                "input_ids": batch.pop('input_ids_2'),
                "attention_mask": batch.pop('attention_mask_2'),
            }
        logits1 = self.feat_head(self.model(*args,**batch)[0][:, 0, :])
        if labels is not None:
            labels = labels.float()
            logits2 = self.feat_head(self.model(**batch2)[0][:, 0, :])
            loss = self.loss_fn([logits1, logits2], labels)
            outputs = (loss,logits1,logits2)
        else:
            outputs = (logits1, )
        return outputs


def evaluate_sample(a_vecs,b_vecs,labels):
    print('*' * 30,'evaluating....',len(a_vecs))
    sims = 1 - paired_distances(a_vecs,b_vecs,metric='cosine')
    print(np.concatenate([sims[:5] , sims[-5:]],axis=0))
    print(np.concatenate([labels[:5] , labels[-5:]],axis=0))
    correlation,_  = stats.spearmanr(labels,sims)
    print('spearman ', correlation)
    return correlation

class MySimpleModelCheckpoint(SimpleModelCheckpoint):
    def __init__(self,*args,**kwargs):
        super(MySimpleModelCheckpoint, self).__init__(*args,**kwargs)
        self.weight_file = './best.pt'

    def on_save_model(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        pl_module: MyTransformer

        #当前设备
        device = torch.device('cuda:{}'.format(trainer.global_rank))
        eval_datasets = dataHelper.load_dataset(dataHelper.eval_files)
        eval_datasets = DataLoader(eval_datasets, batch_size=training_args.eval_batch_size,collate_fn=dataHelper.collate_fn)

        a_vecs, b_vecs, labels = [],[],[]
        for i,batch in tqdm(enumerate(eval_datasets),total=len(eval_datasets),desc='evalute'):
            for k in batch:
                batch[k] = batch[k].to(device)
            o = pl_module.validation_step(batch,i)
            a_logits,b_logits, b_labels = o['outputs']
            for j in range(len(b_logits)):
                logit1 = np.asarray(a_logits[j], dtype=np.float32)
                logit2 = np.asarray(b_logits[j], dtype=np.float32)
                label = np.asarray(b_labels[j], dtype=np.int32)

                a_vecs.append(logit1)
                b_vecs.append(logit2)
                labels.append(label)

        a_vecs = np.stack(a_vecs,axis=0)
        b_vecs = np.stack(b_vecs,axis=0)
        labels =  np.stack(labels,axis=0)
        corrcoef = evaluate_sample(a_vecs, b_vecs,labels)

        f1 = corrcoef
        best_f1 = self.best.get('f1',-np.inf)
        print('current', f1, 'best', best_f1)
        if f1 >= best_f1:
            self.best['f1'] = f1
            logging.info('save best {}, {}\n'.format(self.best['f1'], self.weight_file))
            trainer.save_checkpoint(self.weight_file)


if __name__== '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments,DataArguments))
    model_args, training_args, data_args = parser.parse_dict(train_info_args)

    checkpoint_callback = MySimpleModelCheckpoint(monitor="loss", every_n_epochs=1)
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
    tokenizer, config, label2id, id2label = load_tokenizer_and_config_with_args(dataHelper, model_args, training_args,data_args)

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
                                             with_record_iterable_dataset=True)

    if train_datasets is not None:
        train_datasets = DataLoader(train_datasets, batch_size=training_args.train_batch_size,
                                    collate_fn=dataHelper.collate_fn,
                                    shuffle=False if isinstance(train_datasets, IterableDataset) else True)


    model = MyTransformer(config=config, model_args=model_args, training_args=training_args)

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
            trainer.validate(model, dataloaders=eval_datasets,ckpt_path='./best.pt')

        if test_datasets is not None:
            trainer.test(model, dataloaders=test_datasets,ckpt_path='best.pt')

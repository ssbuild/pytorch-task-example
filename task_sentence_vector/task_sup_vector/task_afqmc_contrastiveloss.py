# -*- coding: utf-8 -*-
import json
import logging
import typing

import numpy as np
import torch
from deep_training.data_helper import DataHelper
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from deep_training.nlp.losses.ContrastiveLoss import ContrastiveLoss
from deep_training.nlp.models.transformer import TransformerModel
from deep_training.utils.func import seq_pading
from deep_training.utils.trainer import SimpleModelCheckpoint
from pytorch_lightning import Trainer
from scipy import stats
from sklearn.metrics.pairwise import paired_distances
from torch import nn
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
    'train_file': [ '/data/nlp/nlp_train_data/senteval_cn/LCQMC/LCQMC.train.data'],
    'eval_file': [ '/data/nlp/nlp_train_data/senteval_cn/LCQMC/LCQMC.valid.data'],
    'test_file': [ '/data/nlp/nlp_train_data/senteval_cn/LCQMC/LCQMC.test.data'],
    'optimizer': 'adamw',
    'learning_rate': 5e-5,
    'max_epochs': 3,
    'train_batch_size': 64,
    'test_batch_size': 2,
    'adam_epsilon': 1e-8,
    'gradient_accumulation_steps': 1,
    'max_grad_norm': 1.0,
    'weight_decay': 0,
    'warmup_steps': 0,
    'output_dir': './output',
    'train_max_seq_length': 64,
    'eval_max_seq_length': 100,
    'test_max_seq_length': 100,
}
# cls , pooler , last-avg , first-last-avg , reduce
pooling = 'cls'


def pad_to_seqlength(sentence, tokenizer, max_seq_length):
    tokenizer: BertTokenizer
    o = tokenizer(sentence, max_length=max_seq_length, truncation=True, add_special_tokens=True,
                  return_token_type_ids=False)
    arrs = [o['input_ids'], o['attention_mask']]
    seqlen = np.asarray(len(arrs[0]), dtype=np.int64)
    input_ids, attention_mask = seq_pading(arrs, max_seq_length=max_seq_length, pad_val=tokenizer.pad_token_id)
    d = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'seqlen': seqlen
    }
    return d


class NN_DataHelper(DataHelper):
    # 切分词
    def on_data_process(self, data: typing.Any, mode: str):
        tokenizer: BertTokenizer
        max_seq_length = self.max_seq_length_dict[mode]
        tokenizer = self.tokenizer
        do_lower_case = tokenizer.do_lower_case
        label2id = self.label2id
        sentence1, sentence2, label_str = data

        d1 = pad_to_seqlength(sentence1, tokenizer, max_seq_length)
        d2 = pad_to_seqlength(sentence2, tokenizer, max_seq_length)
        d = d1
        for k, v in d2.items():
            d[k + '2'] = v

        if label_str is not None:
            labels = np.asarray(int(label_str), dtype=np.int64)
            d['labels'] = labels
        return d

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
                        D.append((jd['sentence1'], jd['sentence2'], jd.get('label', None)))
                else:
                    for line in lines:
                        line = line.replace('\r\n', '').replace('\n', '')
                        s1, s2, l = line.split('\t', 2)
                        D.append((s1, s2, l))
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

        if 'seqlen2' in o:
            seqlen = o.pop('seqlen2')
            max_len = torch.max(seqlen)
            o['input_ids2'] = o['input_ids2'][:, :max_len]
            o['attention_mask2'] = o['attention_mask2'][:, :max_len]
        return o


class MyTransformer(TransformerModel, with_pl=True):
    def __init__(self, *args, **kwargs):
        pooling = kwargs.pop('pooling', 'cls')
        super(MyTransformer, self).__init__(*args, **kwargs)
        self.pooling = pooling
        config = self.config
        self.feat_head = nn.Linear(config.hidden_size, 512, bias=False)
        self.loss_fn = ContrastiveLoss(size_average=False, margin=0.5)

    def get_model_lr(self):
        return super(MyTransformer, self).get_model_lr() + [
            (self.feat_head, self.config.task_specific_params['learning_rate_for_task'])
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
        if labels is not None:
            inputs = {}
            for k in list(batch.keys()):
                if k.endswith('2'):
                    inputs[k.replace('2', '')] = batch.pop(k)
        logits1 = self.forward_for_hidden(*args, **batch)
        if labels is not None:
            labels = torch.squeeze(labels, dim=-1).float()
            logits2 = self.forward_for_hidden(*args, **inputs)
            loss = self.loss_fn([logits1, logits2], labels)
            outputs = (loss, logits1, logits2, labels)
        else:
            outputs = (logits1,)
        return outputs


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

    checkpoint_callback = MySimpleModelCheckpoint(monitor="f1",
                                                  every_n_train_steps=2000 // training_args.gradient_accumulation_steps)
    trainer = Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=training_args.max_epochs,
        max_steps=training_args.max_steps,
        accelerator="gpu",replace_sampler_ddp=False,
        devices=data_args.devices,
        enable_progress_bar=True,
        default_root_dir=data_args.output_dir,
        gradient_clip_val=training_args.max_grad_norm,
        accumulate_grad_batches=training_args.gradient_accumulation_steps,
        num_sanity_val_steps=0,
        strategy='ddp' if torch.cuda.device_count() > 1 else None,
    )

    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer, config, label2id, id2label = dataHelper.load_tokenizer_and_config()

    # 缓存数据集
    if data_args.do_train:
        dataHelper.make_dataset_with_args(data_args.train_file, shuffle=True,mode='train')
    if data_args.do_eval:
        dataHelper.make_dataset_with_args(data_args.eval_file, mode='eval')
    if data_args.do_test:
        dataHelper.make_dataset_with_args(data_args.test_file,mode='test')


    model = MyTransformer(pooling=pooling, config=config, model_args=model_args, training_args=training_args)

    if not data_args.convert_onnx:
        train_datasets = dataHelper.load_dataset(dataHelper.train_files, shuffle=True,infinite=True,
                                                 with_record_iterable_dataset=False,
                                                 with_load_memory=True,num_processes=trainer.world_size,process_index=trainer.global_rank)

        if train_datasets is not None:
            train_datasets = DataLoader(train_datasets, batch_size=training_args.train_batch_size,
                                        collate_fn=dataHelper.collate_fn,
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

# -*- coding: utf-8 -*-
import copy
import json
import random
import typing

import numpy as np
import torch
from deep_training.data_helper import DataHelper
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments, MlmDataArguments
from deep_training.nlp.losses.contrast import SimcseLoss
from deep_training.nlp.models.transformer import TransformerModel
from deep_training.utils.maskedlm import make_mlm_wwm_sample
from fastdatasets.torch_dataset import Dataset as torch_Dataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, IterableDataset
from transformers import HfArgumentParser, BertTokenizer

train_info_args = {
    'devices': '1',
    'data_backend': 'memory_raw',
    'model_type': 'bert',
    'model_name_or_path': '/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'tokenizer_name': '/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'config_name': '/data/nlp/pre_models/torch/bert/bert-base-chinese/config.json',
    'convert_onnx': False, # 转换onnx模型
    'do_train': True, 
    'train_file': [ '/data/nlp/nlp_train_data/thucnews/train.json'],
    'max_epochs': 3,
    'optimizer': 'adamw',
    'learning_rate': 5e-5,
    'train_batch_size': 10,
    'test_batch_size': 2,
    'adam_epsilon': 1e-8,
    'gradient_accumulation_steps': 1,
    'max_grad_norm': 1.0,
    'weight_decay': 0,
    'warmup_steps': 0,
    'output_dir': './output',
    'max_seq_length': 512,
    'do_lower_case': False,
    'do_whole_word_mask': True,
    'max_predictions_per_seq': 20,
    'masked_lm_prob': 0.15
}


class NN_DataHelper(DataHelper):
    # 切分词
    def on_data_process(self, data: typing.Any, mode: str):
        tokenizer: BertTokenizer
        max_seq_length = self.max_seq_length_dict[mode]
        tokenizer = self.tokenizer
        do_lower_case = tokenizer.do_lower_case
        label2id = self.label2id


        rng, do_whole_word_mask, max_predictions_per_seq, masked_lm_prob = self.external_kwargs['mlm_args']

        documents = data
        document_text_string = ''.join(documents)
        document_texts = []
        pos = 0
        while pos < len(document_text_string):
            text = document_text_string[pos:pos + max_seq_length - 2]
            pos += len(text)
            document_texts.append(text)
        # 返回多个文档
        ds = []
        for text in document_texts:
            pair = []
            for _ in range(2):
                node = make_mlm_wwm_sample(text,
                                           tokenizer,
                                           max_seq_length,
                                           rng,
                                           do_whole_word_mask,
                                           max_predictions_per_seq,
                                           masked_lm_prob)
                pair.append(node)
            seqlen = np.max([p.pop('seqlen') for p in pair])
            d = {k: [] for k in pair[0].keys()}
            for node in pair:
                for k, v in node.items():
                    d[k].append(v)
            d = {k: np.stack(v, axis=0) for k, v in d.items()}
            d['seqlen'] = np.asarray(seqlen, dtype=np.int32)
            ds.append(copy.deepcopy(d))
        return ds

    def on_get_corpus(self, files: typing.List, mode: str):
        D = []
        line_no = 0
        for input_file in files:
            with open(input_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    jd = json.loads(line)
                    if not jd:
                        continue
                    text = jd['content']
                    docs = text.split('\n\n')
                    D.append([doc for doc in docs if doc])
                    line_no += 1

                    if line_no > 1000:
                        break

                    if line_no % 10000 == 0:
                        print('read_line', line_no)
                        print(D[-1])
        return D[0:100] if mode == 'train' else D[:10]

    @staticmethod
    def train_collate_fn(batch):
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
        bs, n, s = o['input_ids'].size()
        o = {k: torch.reshape(v, (-1, s)) for k, v in o.items()}
        o['input_ids'] = o['input_ids'][:, :max_len]
        o['attention_mask'] = o['attention_mask'][:, :max_len]
        if 'token_type_ids' in o:
            o['token_type_ids'] = o['token_type_ids'][:, :max_len]
        o['labels'] = o['labels'][:, :max_len]
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
        if 'token_type_ids' in o:
            o['token_type_ids'] = o['token_type_ids'][:, :max_len]
        o['labels'] = o['labels'][:, :max_len]
        return o


class MyTransformer(TransformerModel, with_pl=True):
    def __init__(self, *args, **kwargs):
        super(MyTransformer, self).__init__(*args, **kwargs)
        config = self.config
        self.mlm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.sim_head = nn.Linear(config.hidden_size, 512, bias=False)
        self.loss_fct = CrossEntropyLoss(reduction='mean')
        self.loss_cse = SimcseLoss()

    def get_model_lr(self):
        return super(MyTransformer, self).get_model_lr() + [
            (self.mlm_head, self.config.task_specific_params['learning_rate_for_task']),
            (self.sim_head, self.config.task_specific_params['learning_rate_for_task'])
        ]


    def compute_loss_mlm(self, y_trues, y_preds):
        loss = self.loss_fct(y_preds.view(-1,y_preds.size(-1)), y_trues.view(-1))
        return loss

    def compute_loss(self, *args, **batch) -> tuple:
        labels = None
        if 'labels' in batch:
            labels = batch.pop('labels')

        outputs = self.model(*args, **batch)
        mlm_logits = self.mlm_head(outputs[0])
        simcse_logits = self.sim_head(outputs[1])
        if labels is not None:
            loss1 = self.comput_loss_mlm(labels, mlm_logits)
            loss2 = self.loss_cse(simcse_logits)
            loss = loss1 + loss2
            loss_dict = {
                'mlm_loss': loss1,
                'simcse_loss': loss2,
                'loss': loss
            }
            self.log_dict(loss_dict, prog_bar=True)
            outputs = (loss_dict, mlm_logits, simcse_logits)
        else:
            outputs = (mlm_logits, simcse_logits)
        return outputs


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, MlmDataArguments))
    model_args, training_args, data_args, mlm_data_args = parser.parse_dict(train_info_args)

    checkpoint_callback = ModelCheckpoint(monitor="loss", every_n_train_steps=1000)
    trainer = Trainer(
        log_every_n_steps=20,
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
    rng = random.Random(training_args.seed)
    dataHelper = NN_DataHelper(model_args, training_args, data_args,mlm_args=(rng, mlm_data_args.do_whole_word_mask, mlm_data_args.max_predictions_per_seq,
                  mlm_data_args.masked_lm_prob))
    tokenizer, config, label2id, id2label = dataHelper.load_tokenizer_and_config()

    # 缓存数据集
    if data_args.do_train:
        dataHelper.make_dataset_with_args(data_args.train_file, shuffle=True,mode='train')
    if data_args.do_eval:
        dataHelper.make_dataset_with_args(data_args.eval_file, mode='eval')
    if data_args.do_test:
        dataHelper.make_dataset_with_args(data_args.test_file, shuffle=False,mode='test')

    model = MyTransformer(config=config, model_args=model_args, training_args=training_args)

    if not data_args.convert_onnx:
        train_datasets = dataHelper.load_dataset(dataHelper.train_files, shuffle=True,infinite=True,
                                                 with_record_iterable_dataset=True,num_processes=trainer.world_size,process_index=trainer.global_rank)

        if train_datasets is not None:
            train_datasets = DataLoader(train_datasets, batch_size=training_args.train_batch_size // 2,
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

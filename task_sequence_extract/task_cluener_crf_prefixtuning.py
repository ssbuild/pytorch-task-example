# -*- coding: utf-8 -*-
import json
import logging
import typing

import numpy as np
import torch
from deep_training.data_helper import DataHelper
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments, PrefixModelArguments
from deep_training.data_helper import load_tokenizer_and_config_with_args
from deep_training.nlp.models.prefixtuning import PrefixTransformerForCRF
from deep_training.nlp.models.transformer import TransformerMeta
from deep_training.utils.trainer import CheckpointCallback
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from seqmetric.metrics import f1_score, classification_report
from seqmetric.scheme import IOBES
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import HfArgumentParser, BertTokenizer

train_info_args = {
    'devices': 1,
    'data_backend':'memory_raw',
    'model_type': 'bert',
    'model_name_or_path': '/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'tokenizer_name': '/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'config_name': '/data/nlp/pre_models/torch/bert/bert-base-chinese/config.json',
    'do_train': True,
    'do_eval': True,
    'train_file': '/data/nlp/nlp_train_data/clue/cluener/train.json',
    'eval_file': '/data/nlp/nlp_train_data/clue/cluener/dev.json',
    'test_file': '/data/nlp/nlp_train_data/clue/cluener/test.json',
    'optimizer': 'adamw',
    'learning_rate': 1e-3,
    'learning_rate_for_task': 1e-4,
    'max_epochs': 3,
    'train_batch_size': 10,
    'eval_batch_size': 2,
    'test_batch_size': 2,
    'adam_epsilon': 1e-8,
    'gradient_accumulation_steps': 1,
    'max_grad_norm': 1.0,
    'weight_decay': 0,
    'warmup_steps': 0,
    'output_dir': './output',
    'max_seq_length': 160,
    'pre_seq_len': 100
}

class NN_DataHelper(DataHelper):
    index = -1
    def on_data_ready(self):
        self.index = -1
    # 切分词
    def on_data_process(self, data: typing.Any, user_data: tuple):
        self.index += 1

        tokenizer: BertTokenizer
        tokenizer, max_seq_length, do_lower_case, label2id, mode = user_data
        sentence, label_dict = data

        tokens = list(sentence) if not do_lower_case else list(sentence.lower())
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        if len(input_ids) > max_seq_length - 2:
            input_ids = input_ids[:max_seq_length - 2]
        input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)

        input_ids = np.asarray(input_ids, dtype=np.int64)
        attention_mask = np.asarray(attention_mask, dtype=np.int64)
        seqlen = np.asarray(len(input_ids), dtype=np.int64)

        labels = np.zeros(shape=(seqlen,), dtype=np.int64)
        for label_str, o in label_dict.items():
            pts = [_ for a_ in list(o.values()) for _ in a_]
            for pt in pts:
                if pt[1] > seqlen - 2:
                    continue
                pt[0] += 1
                pt[1] += 1
                span_len = pt[1] - pt[0] + 1
                if span_len == 1:
                    labels[pt[0]] = label2id['S-' + label_str]
                elif span_len >= 2:
                    labels[pt[0]] = label2id['B-' + label_str]
                    labels[pt[1]] = label2id['E-' + label_str]
                    for i in range(span_len - 2):
                        labels[pt[0] + 1 + i] = label2id['I-' + label_str]

        pad_len = max_seq_length - len(input_ids)
        if pad_len > 0:
            pad_val = tokenizer.pad_token_id
            input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            attention_mask = np.pad(attention_mask, (0, pad_len), 'constant', constant_values=(0, 0))
            labels = np.pad(labels, (0, pad_len), 'constant', constant_values=(0, 0))

        d = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'seqlen': seqlen,
        }

        if self.index < 5:
            print(tokens)
            print(input_ids[:seqlen])
            print(attention_mask[:seqlen])
            print(labels[:seqlen])
            print(seqlen)
        return d

    #读取标签
    def on_get_labels(self, files: typing.List[str]):
        labels = [
            'address','book','company','game','government','movie','name','organization','position','scene'
        ]
        labels = ['O'] + [t + '-' + l  for t in ['B','I','E','S'] for l in labels]
        label2id = {label: i for i, label in enumerate(labels)}
        id2label = {i: label for i, label in enumerate(labels)}
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
                    D.append((jd['text'], jd.get('label',None)))
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

        o['labels'] = o['labels'][:,:max_len]
        return o

class MyTransformer(PrefixTransformerForCRF, metaclass=TransformerMeta):
    def __init__(self, *args,**kwargs):
        super(MyTransformer, self).__init__(*args,**kwargs)

    def validation_epoch_end(self, outputs: typing.Union[EPOCH_OUTPUT, typing.List[EPOCH_OUTPUT]]) -> None:
        y_preds, y_trues = [], []
        for o in outputs:
            preds, labels = o['outputs']
            for p, l in zip(preds, labels):
                y_preds.append(p)
                y_trues.append(l)

        label_map = self.config.id2label
        trues_list = [[] for _ in range(len(y_trues))]
        preds_list = [[] for _ in range(len(y_preds))]

        for i in range(len(y_trues)):
            for j in range(len(y_trues[i])):
                if y_trues[i][j] != self.config.pad_token_id:
                    trues_list[i].append(label_map[y_trues[i][j]])
                    preds_list[i].append(label_map[y_preds[i][j]])

        scheme = IOBES
        f1 = f1_score(trues_list, preds_list, average='macro', scheme=scheme)
        report = classification_report(trues_list, preds_list, scheme=scheme, digits=4)

        print(f1, report)
        self.log('val_f1', f1)


class MyCheckpointCallback(CheckpointCallback):
    def __init__(self,*args,**kwargs):
        super(MyCheckpointCallback, self).__init__(*args,**kwargs)
        self.weight_file = './best.pt'

    def on_save_model(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        pl_module: MyTransformer

        #当前设备
        device = torch.device('cuda:{}'.format(trainer.global_rank))
        eval_datasets = dataHelper.load_dataset(dataHelper.eval_files)
        eval_datasets = DataLoader(eval_datasets, batch_size=training_args.eval_batch_size,collate_fn=dataHelper.collate_fn)

        threshold = 1e-8
        eval_labels = pl_module.eval_labels
        config = pl_module.config


        y_preds, y_trues = [], []
        for i,batch in tqdm(enumerate(eval_datasets),total=len(eval_datasets),desc='evalute'):
            for k in batch:
                batch[k] = batch[k].to(device)
            o = pl_module.validation_step(batch,i)

            preds, labels = o['outputs']
            for p, l in zip(preds, labels):
                y_preds.append(p)
                y_trues.append(l)

        label_map = config.id2label
        trues_list = [[] for _ in range(len(y_trues))]
        preds_list = [[] for _ in range(len(y_preds))]

        for i in range(len(y_trues)):
            for j in range(len(y_trues[i])):
                if y_trues[i][j] != config.pad_token_id:
                    trues_list[i].append(label_map[y_trues[i][j]])
                    preds_list[i].append(label_map[y_preds[i][j]])

        scheme = IOBES
        f1 = f1_score(trues_list, preds_list, average='macro', scheme=scheme)
        report = classification_report(trues_list, preds_list, scheme=scheme, digits=4)
        print(f1, report)


        if not hasattr(self.best, 'f1'):
            self.best['f1'] = f1
        print('current', f1, 'best', self.best['f1'])
        if f1 >= self.best['f1']:
            self.best['f1'] = f1
            logging.info('save best {}, {}\n'.format(self.best['f1'], self.weight_file))
            trainer.save_checkpoint(self.weight_file)



if __name__== '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments,PrefixModelArguments))
    model_args, training_args, data_args,prompt_args = parser.parse_dict(train_info_args)

    checkpoint_callback = MyCheckpointCallback(monitor='val_f1', every_n_epochs=1)
    trainer = Trainer(
        log_every_n_steps=10,
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
        dataHelper.train_files.append(dataHelper.make_dataset_with_args(data_args.train_file, token_fn_args_dict['train'],
                                                                   data_args,
                                                                   intermediate_name=intermediate_name, shuffle=True,
                                                                   mode='train')) 
    if data_args.do_eval:
        dataHelper.eval_files.append(dataHelper.make_dataset_with_args(data_args.eval_file, token_fn_args_dict['eval'],
                                                                  data_args,
                                                                  intermediate_name=intermediate_name, shuffle=False,
                                                                  mode='eval')) 
    if data_args.do_test:
        dataHelper.test_files.append(dataHelper.make_dataset_with_args(data_args.test_file, token_fn_args_dict['test'],
                                                                  data_args,
                                                                  intermediate_name=intermediate_name, shuffle=False,
                                                                  mode='test'))

    train_datasets = dataHelper.load_dataset(dataHelper.train_files, shuffle=True, num_processes=trainer.world_size,
                                             process_index=trainer.global_rank, infinite=True,
                                             with_record_iterable_dataset=True)
    if train_datasets is not None:
        train_datasets = DataLoader(train_datasets, batch_size=training_args.train_batch_size,
                                        collate_fn=dataHelper.collate_fn,
                                        shuffle=False if isinstance(train_datasets, IterableDataset) else True)


    model = MyTransformer(prompt_args=prompt_args,config=config,model_args=model_args,training_args=training_args)


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
            trainer.validate(model, dataloaders=eval_datasets)

        if test_datasets is not None:
            trainer.test(model, dataloaders=test_datasets)

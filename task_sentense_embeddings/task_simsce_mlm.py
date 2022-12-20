# -*- coding: utf-8 -*-
import json
import random
import typing

import torch
from deep_training.data_helper import DataHelper
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments, MlmDataArguments
from deep_training.data_helper import load_tokenizer_and_config_with_args
from deep_training.nlp.losses.contrast import compute_simcse_loss
from deep_training.nlp.models.transformer import TransformerModel, TransformerMeta
from deep_training.utils.maskedlm import make_mlm_wwm_sample
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, IterableDataset
from transformers import HfArgumentParser, BertTokenizer

train_info_args = {
    'devices':'1',
    'data_backend': 'memory_raw',
    'model_type': 'bert',
    'model_name_or_path': '/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'tokenizer_name': '/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'config_name': '/data/nlp/pre_models/torch/bert/bert-base-chinese/config.json',
    'do_train': True,
    'train_file': '/data/nlp/nlp_train_data/thucnews/train.json',
    'max_steps': 100000,
    'optimizer': 'adamw',
    'learning_rate': 5e-5,
    'train_batch_size': 10,
    'test_batch_size': 2,
    'adam_epsilon': 1e-8,
    'gradient_accumulation_steps': 1,
    'max_grad_norm': 1.0,
    'weight_decay': 0,
    'warmup_steps': 0,
    'output_dir' : './output',
    'max_seq_length' : 512,
    'do_lower_case': False,
    'do_whole_word_mask': True,
    'max_predictions_per_seq': 20,
    'masked_lm_prob': 0.15
}


class NN_DataHelper(DataHelper):
    # 切分词
    def on_data_process(self, data: typing.Any, user_data: tuple):
        tokenizer: BertTokenizer
        tokenizer,max_seq_length, do_lower_case, label2id,\
        rng, do_whole_word_mask, max_predictions_per_seq, masked_lm_prob,mode = user_data
        # assert isinstance(data,tuple)
        documents = data
        document_text_string = ''.join(documents)
        document_texts = []
        pos = 0
        while pos < len(document_text_string):
            text = document_text_string[pos:pos + max_seq_length - 2]
            pos += len(text)
            document_texts.append(text)
        # 返回多个文档
        document_nodes = []
        for text in document_texts:
            for _ in range(2):
                node = make_mlm_wwm_sample(text,
                                           tokenizer,
                                           max_seq_length,
                                           rng,
                                           do_whole_word_mask,
                                           max_predictions_per_seq,
                                           masked_lm_prob)
                document_nodes.append(node)
        return document_nodes

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
        o['labels'] = o['labels'][:, :max_len]
        o['weight'] = o['weight'][:, :max_len]
        return o

class MyTransformer(TransformerModel, metaclass=TransformerMeta):
    def __init__(self,*args,**kwargs):
        super(MyTransformer, self).__init__(*args,**kwargs)
        config = self.config
        self.mlm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.sim_head = nn.Linear(config.hidden_size, 512, bias=False)
        self.loss_fct = CrossEntropyLoss(reduction='none', ignore_index=self.config.pad_token_id)

    def get_model_lr(self):
        return super(MyTransformer, self).get_model_lr() + [
            (self.mlm_head, self.config.task_specific_params['learning_rate_for_task']),
            (self.sim_head, self.config.task_specific_params['learning_rate_for_task'])
        ]

    def comput_loss_mlm(self,y_trues, y_preds, weight):
        y_preds = torch.transpose(y_preds, 1, 2)
        loss = self.loss_fct(y_preds, y_trues)
        loss = loss * weight
        loss = torch.sum(loss, dtype=torch.float) / (torch.sum(weight, dtype=torch.float) + 1e-8)
        return loss

    def compute_loss(self, batch,batch_idx):
        labels,weight = None,None
        if 'labels' in batch:
            labels = batch.pop('labels')
            weight = batch.pop('weight')

        outputs = self(**batch)
        mlm_logits = self.mlm_head(outputs[0])
        simcse_logits = self.sim_head(outputs[1])
        if labels is not None:
            loss1 = self.comput_loss_mlm(labels, mlm_logits, weight)
            loss2 = compute_simcse_loss(simcse_logits)
            loss = loss1 + loss2
            loss_dict = {
                'mlm_loss': loss1,
                'simcse_loss': loss2,
                'loss': loss
            }
            self.log_dict(loss_dict,prog_bar=True)
            outputs = (loss_dict,mlm_logits,simcse_logits)
        else:
            outputs = (mlm_logits,simcse_logits)
        return outputs




def get_trainer():
    checkpoint_callback = ModelCheckpoint(monitor="loss", save_last=False, every_n_epochs=1)
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
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments,MlmDataArguments))
    model_args, training_args, data_args,mlm_data_args = parser.parse_dict(train_info_args)


    trainer = get_trainer()
    dataHelper = NN_DataHelper(data_args.data_backend)
    tokenizer, config, label2id, id2label = load_tokenizer_and_config_with_args(dataHelper, model_args, training_args,data_args)
    rng = random.Random(training_args.seed)


    token_fn_args_dict = {
        'train': (tokenizer, data_args.train_max_seq_length, model_args.do_lower_case, label2id,
                  rng, mlm_data_args.do_whole_word_mask, mlm_data_args.max_predictions_per_seq,
                  mlm_data_args.masked_lm_prob,'train'),
        'eval': (tokenizer, data_args.eval_max_seq_length, model_args.do_lower_case, label2id,
                 rng, mlm_data_args.do_whole_word_mask, mlm_data_args.max_predictions_per_seq,
                 mlm_data_args.masked_lm_prob,
                 'eval'),
        'test': (tokenizer, data_args.test_max_seq_length, model_args.do_lower_case, label2id,
                 rng, mlm_data_args.do_whole_word_mask, mlm_data_args.max_predictions_per_seq,
                 mlm_data_args.masked_lm_prob,
                 'test')
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

    train_datasets = dataHelper.load_dataset(train_files, shuffle=False)
    eval_datasets = dataHelper.load_dataset(eval_files)
    test_datasets = dataHelper.load_dataset(test_files)
    if train_datasets is not None:
        train_datasets = DataLoader(train_datasets, batch_size=training_args.train_batch_size,
                                    collate_fn=dataHelper.collate_fn,
                                    shuffle=False if isinstance(train_datasets, IterableDataset) else False)
    if eval_datasets is not None:
        eval_datasets = DataLoader(eval_datasets, batch_size=training_args.eval_batch_size,
                                   collate_fn=dataHelper.collate_fn)
    if test_datasets is not None:
        test_datasets = DataLoader(test_datasets, batch_size=training_args.test_batch_size,
                                   collate_fn=dataHelper.collate_fn)
    

    model = MyTransformer(config=config,model_args=model_args,training_args=training_args)

    if train_datasets is not None:
        trainer.fit(model, train_dataloaders=train_datasets, val_dataloaders=eval_datasets)

    if eval_datasets is not None:
        trainer.validate(model, dataloaders=eval_datasets)

    if test_datasets is not None:
        trainer.test(model, dataloaders=test_datasets)

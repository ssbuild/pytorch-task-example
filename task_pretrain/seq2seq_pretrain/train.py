# -*- coding: utf-8 -*-
import json
import typing

import numpy as np
import torch
from deep_training.data_helper import DataHelper
from deep_training.data_helper import ModelArguments, DataArguments, TrainingArguments
from deep_training.nlp.models.transformer import TransformerForSeq2SeqLM
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, IterableDataset
from transformers import HfArgumentParser, BertTokenizer

train_info_args = {
    'devices': 1,
    'data_backend': 'memory_raw',
    'model_type': 't5',
    # 'model_name_or_path': '/data/nlp/pre_models/torch/',
    'tokenizer_name': './t5_small_config',
    'config_name': './t5_small_config/config.json',
    'do_train': True,
    'train_file': '/data/nlp/nlp_train_data/thucnews/train.json',
    'learning_rate': 5e-5,
    'max_epochs': 3,
    'train_batch_size': 10,
    'test_batch_size': 2,
    'adam_epsilon': 1e-8,
    'gradient_accumulation_steps': 1,
    'max_grad_norm': 1.0,
    'weight_decay': 0,
    'warmup_steps': 0,
    'output_dir': './output',
    'train_max_seq_length': 512,
    'eval_max_seq_length': 512,
    'test_max_seq_length': 512,
    'max_target_length': 64,
}


class NN_DataHelper(DataHelper):
    # 切分词
    def on_data_process(self, data: typing.Any, mode: str):
        tokenizer: BertTokenizer
        max_seq_length = self.max_seq_length_dict[mode]
        tokenizer = self.tokenizer
       

        max_target_length = self.data_args.max_target_length


        x = data
        o1 = tokenizer.encode_plus(x[0], max_length=max_seq_length, truncation=True, add_special_tokens=True, )
        o2 = tokenizer.encode_plus(x[1], max_length=max_target_length, truncation=True, add_special_tokens=True, )
        input_ids = np.asarray(o1['input_ids'], dtype=np.int64)
        attention_mask = np.asarray(o1['attention_mask'], dtype=np.int64)

        slen = np.asarray(len(input_ids), dtype=np.int64)
        pad_len = max_seq_length - slen
        if pad_len > 0:
            pad_val = tokenizer.pad_token_id
            input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            attention_mask = np.pad(attention_mask, (0, pad_len), 'constant', constant_values=(0, 0))

        decoder_input_ids = np.asarray(o2['input_ids'], dtype=np.int64)
        labels = np.asarray(decoder_input_ids[1:], dtype=np.int64)
        decoder_input_ids = np.asarray(decoder_input_ids[:-1], dtype=np.int64)
        decoder_attention_mask = np.asarray([1] * len(decoder_input_ids), dtype=np.int64)

        dlen = np.asarray(len(decoder_input_ids), dtype=np.int64)
        pad_len = max_target_length - dlen
        if pad_len > 0:
            pad_val = tokenizer.pad_token_id
            decoder_input_ids = np.pad(decoder_input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            decoder_attention_mask = np.pad(decoder_attention_mask, (0, pad_len), 'constant', constant_values=(0, 0))
            labels = np.pad(labels, (0, pad_len+1), 'constant', constant_values=(0, 0))

        d = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'decoder_input_ids': decoder_input_ids,
            'decoder_attention_mask': decoder_attention_mask,
            'labels': labels,
            'slen': slen,
            'dlen': dlen
        }
        return d

    # 读取文件
    def on_get_corpus(self, files: typing.List, mode: str):
        D = []
        for filename in files:
            with open(filename, mode='r', encoding='utf-8') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    jd = json.loads(line)
                    D.append((jd['content'], jd['title']))
                    if i > 1000:
                        break
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

        slen = torch.max(o.pop('slen'))
        dlen = torch.max(o.pop('dlen'))

        o['input_ids'] = o['input_ids'][:, :slen]
        o['attention_mask'] = o['attention_mask'][:, :slen]

        o['decoder_input_ids'] = o['decoder_input_ids'][:, :dlen]
        o['decoder_attention_mask'] = o['decoder_attention_mask'][:, :dlen]
        o['labels'] = o['labels'][:, :dlen]
        return o


class MyTransformer(TransformerForSeq2SeqLM, with_pl=True):
    def __init__(self, *args, **kwargs):
        super(MyTransformer, self).__init__(*args, **kwargs)
        self.loss_fct = CrossEntropyLoss(ignore_index=self.config.pad_token_id)

    def compute_loss(self, *args, **batch) -> tuple:
        labels = batch.pop('labels', None)
        outputs = self.model(*args, **batch)
        lm_logits = outputs[0]
        if labels is not None:
            loss = self.loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            outputs = (loss, lm_logits, labels)
        else:
            outputs = (lm_logits,)
        return outputs


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
    model_args, training_args, data_args = parser.parse_dict(train_info_args)

    checkpoint_callback = ModelCheckpoint(monitor="loss", save_top_k=5,
                                          every_n_train_steps=2000 // training_args.gradient_accumulation_steps)
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
    tokenizer, config, label2id, id2label = dataHelper.load_tokenizer_and_config(model_args, training_args, data_args)


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
            trainer.validate(model, dataloaders=eval_datasets, ckpt_path='./best.pt')

        if test_datasets is not None:
            trainer.test(model, dataloaders=test_datasets, ckpt_path='best.pt')

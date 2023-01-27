# -*- coding: utf-8 -*-
import json
import logging
import typing

import numpy as np
import torch
from deep_training.data_helper import DataHelper
from deep_training.data_helper import ModelArguments, DataArguments, TrainingArguments
from deep_training.nlp.metrics.pointer import metric_for_pointer
from deep_training.nlp.models.w2ner import TransformerForW2ner, extract_lse, W2nerArguments
from deep_training.utils.trainer import SimpleModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
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
    'train_file': [ '/data/nlp/nlp_train_data/clue/cluener/train.json'],
    'eval_file': [ '/data/nlp/nlp_train_data/clue/cluener/dev.json'],
    'test_file': [ '/data/nlp/nlp_train_data/clue/cluener/test.json'],
    'optimizer': 'adamw',
    'learning_rate': 5e-5,
    'learning_rate_for_task': 5e-5,
    'max_epochs': 15,
    'train_batch_size': 40,
    'eval_batch_size': 2,
    'test_batch_size': 1,
    'adam_epsilon': 1e-8,
    'gradient_accumulation_steps': 1,
    'max_grad_norm': 1.0,
    'weight_decay': 0,
    'warmup_steps': 0,
    'output_dir': './output',
    'train_max_seq_length': 90,
    'eval_max_seq_length': 120,
    'test_max_seq_length': 120,
    # w2nerArguments
    'use_last_4_layers': False,  # 例如bert 后四层平均
    'dist_emb_size': 20,
    'type_emb_size': 20,
    'lstm_hid_size': 768,
    'conv_hid_size': 96,
    'biaffine_size': 768,
    'ffnn_hid_size': 768,
    'dilation': [1, 2, 3],
    'emb_dropout': 0.1,
    'conv_dropout': 0.1,
    'out_dropout': 0.1,

}


class NN_DataHelper(DataHelper):
    index = -1
    eval_labels = []

    def __init__(self, *args, **kwargs):
        super(NN_DataHelper, self).__init__(*args, **kwargs)
        dis2idx = np.zeros((1000), dtype='int64')
        dis2idx[1] = 1
        dis2idx[2:] = 2
        dis2idx[4:] = 3
        dis2idx[8:] = 4
        dis2idx[16:] = 5
        dis2idx[32:] = 6
        dis2idx[64:] = 7
        dis2idx[128:] = 8
        dis2idx[256:] = 9

        self.dis2idx = dis2idx

    # 切分成开始
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
        sentence, entities = data

        tokens = list(sentence) if not do_lower_case else list(sentence.lower())
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        if len(input_ids) > max_seq_length - 2:
            input_ids = input_ids[:max_seq_length - 2]
        input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)

        input_ids = np.asarray(input_ids, dtype=np.int32)
        attention_mask = np.asarray(attention_mask, dtype=np.int32)
        seqlen = np.asarray(len(input_ids), dtype=np.int32)

        real_label = []
        length = len(input_ids)
        grid_labels = np.zeros((length, length), dtype=np.int32)
        pieces2word = np.zeros((length, length), dtype=bool)
        dist_inputs = np.zeros((length, length), dtype=np.int32)
        grid_mask2d = np.ones((length, length), dtype=bool)

        for i in range(seqlen - 1):
            for j in range(i + 1, i + 2):
                pieces2word[i][j] = 1
        for k in range(seqlen):
            dist_inputs[k, :] += k
            dist_inputs[:, k] -= k

        for i in range(length):
            for j in range(length):
                if dist_inputs[i, j] < 0:
                    dist_inputs[i, j] = self.dis2idx[-dist_inputs[i, j]] + 9
                else:
                    dist_inputs[i, j] = self.dis2idx[dist_inputs[i, j]]
        dist_inputs[dist_inputs == 0] = 19

        if entities is not None:
            for l, s, e in entities:
                l = label2id[l]
                real_label.append((l, s, e))
                s += 1
                e += 1
                if s < max_seq_length - 1 and e < max_seq_length - 1:
                    for i in range(e - s):
                        grid_labels[i, i + 1] = 1
                    grid_labels[e, s] = l + 2

        pad_len = max_seq_length - len(input_ids)
        if pad_len > 0:
            input_ids = np.pad(input_ids, (0, pad_len), 'constant',
                               constant_values=(tokenizer.pad_token_id, tokenizer.pad_token_id))
            attention_mask = np.pad(attention_mask, (0, pad_len), 'constant', constant_values=(0, 0))

            grid_labels = np.pad(grid_labels, pad_width=((0, pad_len), (0, pad_len)))
            pieces2word = np.pad(pieces2word, pad_width=((0, pad_len), (0, pad_len)))
            dist_inputs = np.pad(dist_inputs, pad_width=((0, pad_len), (0, pad_len)))
            grid_mask2d = np.pad(grid_mask2d, pad_width=((0, pad_len), (0, pad_len)))

        d = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': grid_labels,
            'pieces2word': pieces2word,
            'dist_inputs': dist_inputs,
            'grid_mask2d': grid_mask2d,
            'seqlen': seqlen,
        }

        # if self.index < 5:
        #     print(tokens)
        #     print(input_ids[:seqlen])
        #     print(attention_mask[:seqlen])
        #     print(seqlen)

        if mode == 'eval':
            if self.index < 3:
                print(sentence, entities)
            self.eval_labels.append(real_label)
        return d

    # 读取标签
    def on_get_labels(self, files: typing.List[str]):
        labels = [
            'address', 'book', 'company', 'game', 'government', 'movie', 'name', 'organization', 'position', 'scene'
        ]
        labels = list(set(labels))
        labels = sorted(labels)
        label2id = {label: i for i, label in enumerate(labels)}
        id2label = {i: label for i, label in enumerate(labels)}
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
                    # cluener 为 label，  fastlabel 为 entities
                    entities = jd.get('label', None)
                    if entities:
                        entities_label = []
                        for k, v in entities.items():
                            pts = [_ for a_ in list(v.values()) for _ in a_]
                            for pt in pts:
                                assert pt[0] <= pt[1], ValueError(line, pt)
                                entities_label.append((k, pt[0], pt[1]))
                    else:
                        entities_label = None
                    D.append((jd['text'], entities_label))
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
        o['labels'] = o['labels'][:, :max_len, :max_len]
        o['pieces2word'] = o['pieces2word'][:, :max_len, :max_len]
        o['dist_inputs'] = o['dist_inputs'][:, :max_len, :max_len]
        o['grid_mask2d'] = o['grid_mask2d'][:, :max_len, :max_len]

        return o


class MyTransformer(TransformerForW2ner, with_pl=True):
    def __init__(self, eval_labels, *args, **kwargs):
        super(MyTransformer, self).__init__(*args, **kwargs)
        self.model.eval_labels = eval_labels
        self.eval_labels = eval_labels

    def validation_epoch_end(self, outputs: typing.Union[EPOCH_OUTPUT, typing.List[EPOCH_OUTPUT]]) -> None:
        label2id = self.config.label2id
        y_preds, y_trues = [], []
        eval_labels = self.eval_labels
        for i, o in enumerate(outputs):
            logits, seqlens, _ = o['outputs']
            y_preds.extend(extract_lse([logits, seqlens]))
            bs = len(logits)
            y_trues.extend(eval_labels[i * bs: (i + 1) * bs])

        print(y_preds[:3])
        print(y_trues[:3])

        f1, str_report = metric_for_pointer(y_trues, y_preds, label2id)
        print(f1)
        print(str_report)
        self.log('val_f1', f1, prog_bar=True)


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

        eval_labels = pl_module.eval_labels
        config = pl_module.config

        y_preds, y_trues = [], []
        for i, batch in tqdm(enumerate(eval_datasets), total=len(eval_datasets), desc='evalute'):
            for k in batch:
                batch[k] = batch[k].to(device)
            o = pl_module.validation_step(batch, i)

            logits, seqlens, _ = o['outputs']
            y_preds.extend(extract_lse([logits, seqlens]))
            bs = len(logits)
            y_trues.extend(eval_labels[i * bs: (i + 1) * bs])

        print(y_preds[:3])
        print(y_trues[:3])

        f1, str_report = metric_for_pointer(y_trues, y_preds, config.label2id)
        print(f1)
        print(str_report)

        best_f1 = self.best.get('f1', -np.inf)
        print('current', f1, 'best', best_f1)
        if f1 >= best_f1:
            self.best['f1'] = f1
            logging.info('save best {}, {}\n'.format(self.best['f1'], self.weight_file))
            trainer.save_checkpoint(self.weight_file)


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, W2nerArguments))
    model_args, training_args, data_args, w2nerArguments = parser.parse_dict(train_info_args)

    checkpoint_callback = MySimpleModelCheckpoint(monitor='val_f1', every_n_epochs=1)
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

    model = MyTransformer(dataHelper.eval_labels, w2nerArguments=w2nerArguments, config=config, model_args=model_args,
                          training_args=training_args)

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

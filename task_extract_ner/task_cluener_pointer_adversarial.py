# -*- coding: utf-8 -*-
# 对抗训练
import json
import logging
import typing

import numpy as np
import torch
from deep_training.data_helper import DataHelper
from deep_training.data_helper import ModelArguments, DataArguments, TrainingArguments
from deep_training.nlp.layers.seq_pointer import f1_metric_for_pointer
from deep_training.nlp.losses.loss_globalpointer import loss_for_pointer
from deep_training.nlp.metrics.pointer import metric_for_pointer
from deep_training.nlp.models.pointer import TransformerForPointer, extract_lse
from deep_training.utils.trainer import SimpleModelCheckpoint
from pytorch_lightning import Trainer
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
    'convert_onnx': False, # 转换onnx模型
    'do_train': True, 
    'do_eval': True,
    'train_file': [ '/data/nlp/nlp_train_data/clue/cluener/train.json'],
    'eval_file': [ '/data/nlp/nlp_train_data/clue/cluener/dev.json'],
    'test_file': [ '/data/nlp/nlp_train_data/clue/cluener/test.json'],
    'learning_rate': 5e-5,
    'learning_rate_for_task': 5e-5,
    'max_epochs': 15,
    'train_batch_size': 40,
    'eval_batch_size': 1,
    'test_batch_size': 1,
    'adam_epsilon': 1e-8,
    'gradient_accumulation_steps': 1,
    'max_grad_norm': 1.0,
    'weight_decay': 0,
    'warmup_steps': 0,
    'output_dir': './output',
    'train_max_seq_length': 380,
    'eval_max_seq_length': 512,
    'test_max_seq_length': 512,
    # 对抗训练就一个配置
    'adv': {
        'mode': 'fgm',  # None, fgm, fgsm_local, fgsm(不推荐), pgd, free_local, free(不推荐)
        'emb_name': 'embedding',
        'attack_iters': 2,  # pgd
        'minibatch_replays': 2,  # free
        'alpha': 0.5,  # pgd,fgsm
        'epsilon': 0.5,  # pgd,fgm
    }
}


class NN_DataHelper(DataHelper):
    index = -1
    eval_labels = []

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
        labels = np.zeros(shape=(len(label2id), max_seq_length, max_seq_length), dtype=np.int32)
        real_label = []

        if entities is not None:
            for l, s, e in entities:
                l = label2id[l]
                real_label.append((l, s, e))
                s += 1
                e += 1
                if s < max_seq_length - 1 and e < max_seq_length - 1:
                    labels[l, s, e] = 1

        pad_len = max_seq_length - len(input_ids)
        if pad_len > 0:
            input_ids = np.pad(input_ids, (0, pad_len), 'constant',
                               constant_values=(tokenizer.pad_token_id, tokenizer.pad_token_id))
            attention_mask = np.pad(attention_mask, (0, pad_len), 'constant', constant_values=(0, 0))
        d = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'seqlen': seqlen,
        }

        # if self.index < 5:
        #     print(tokens)
        #     print(input_ids[:seqlen])
        #     print(attention_mask[:seqlen])
        #     print(seqlen)

        if mode == 'eval':
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
        o['labels'] = o['labels'][:, :, :max_len, :max_len]
        return o


class MyTransformer(TransformerForPointer, with_pl=True):
    def __init__(self, eval_labels, *args, **kwargs):
        super(MyTransformer, self).__init__(*args, **kwargs)
        self.model.eval_labels = eval_labels
        self.eval_labels = eval_labels

    def compute_loss(self, *args, **batch) -> tuple:
        labels: torch.Tensor = batch.pop('labels', None)
        outputs = self.model(*args, **batch)
        logits = outputs[0]
        if self.model.training:
            logits = self.model.dropout(logits)
        logits = self.model.pointer_layer(logits, batch['attention_mask'] if len(batch) else args[1])
        if labels is not None:
            loss = loss_for_pointer(labels, logits)
            f1 = f1_metric_for_pointer(labels, logits)
            loss_dict = {'loss': loss, 'f1': f1}
            outputs = (loss_dict, logits, labels)
        else:
            outputs = (logits,)
        return outputs


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

        threshold = 1e-8
        eval_labels = pl_module.eval_labels
        config = pl_module.config

        y_preds, y_trues = [], []
        for i, batch in tqdm(enumerate(eval_datasets), total=len(eval_datasets), desc='evalute'):
            for k in batch:
                batch[k] = batch[k].to(device)
            o = pl_module.validation_step(batch, i)

            logits, _ = o['outputs']
            y_preds.extend(extract_lse(logits, threshold))
            bs = len(logits)
            y_trues.extend(eval_labels[i * bs: (i + 1) * bs])

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
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
    model_args, training_args, data_args = parser.parse_dict(train_info_args)

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

    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer, config, label2id, id2label = dataHelper.load_tokenizer_and_config()

    # 缓存数据集
    if data_args.do_train:
        dataHelper.make_dataset_with_args(data_args.train_file, shuffle=True,mode='train')
    if data_args.do_eval:
        dataHelper.make_dataset_with_args(data_args.eval_file,shuffle=False, mode='eval')
    if data_args.do_test:
        dataHelper.make_dataset_with_args(data_args.test_file,shuffle=False,mode='test')

    model = MyTransformer(dataHelper.eval_labels, with_efficient=True, config=config, model_args=model_args,
                          training_args=training_args)

    if not data_args.convert_onnx:
        train_datasets = dataHelper.load_dataset(dataHelper.train_files, shuffle=True, num_processes=trainer.world_size,
                                                 process_index=trainer.global_rank, infinite=True,
                                                 with_record_iterable_dataset=True)
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

            is_convert_onnx = True
            # 是否转换模型
            if is_convert_onnx:
                model.convert_to_onnx('./best.onnx')

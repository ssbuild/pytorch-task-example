# -*- coding: utf-8 -*-
import json
import typing

import numpy as np
import torch
from deep_training.data_helper import DataHelper
from deep_training.data_helper import ModelArguments, DataArguments, TrainingArguments
from deep_training.nlp.layers.mask import unilm_mask
from deep_training.nlp.losses.loss_kl import KLDivLoss
from deep_training.nlp.models.transformer import TransformerModelForUnilm
from deep_training.utils.func import seq_padding
from deep_training.utils.trainer import SimpleModelCheckpoint
from lightning import Trainer
from torch.utils.data import DataLoader, IterableDataset
from transformers import BertTokenizer
from transformers import HfArgumentParser

train_info_args = {
    'devices': 1,
    'data_backend': 'memory_raw',
    'model_type': 'bert',
    'model_name_or_path': '/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'tokenizer_name': '/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'config_name': '/data/nlp/pre_models/torch/bert/bert-base-chinese/config.json',
    'convert_onnx': False, # 转换onnx模型
    'do_train': True, 
    'train_file': [ '/data/nlp/nlp_train_data/thucnews/train.json'],
    'max_steps': 100000,
    'train_batch_size': 8,
    'test_batch_size': 2,
    'adam_epsilon': 1e-8,
    'gradient_accumulation_steps': 1,
    'max_grad_norm': 1.0,
    'weight_decay': 0,
    'warmup_steps': 0,
    'output_dir': './output',
    'max_seq_length': 200,
    'max_target_length': 50
}


class NN_DataHelper(DataHelper):
    # 切分词
    def on_data_process(self, data: typing.Any, mode: str):
        tokenizer: BertTokenizer
        max_seq_length = self.max_seq_length_dict[mode]
        tokenizer = self.tokenizer
        do_lower_case = tokenizer.do_lower_case
        label2id = self.label2id
        x = data
        assert isinstance(x, tuple)
        o = tokenizer.encode_plus(text=x[0], text_pair=x[1], max_length=max_seq_length, truncation=True)
        seqlen = np.asarray(len(o['input_ids']), dtype=np.int32)
        input_ids = seq_padding(o['input_ids'], max_seq_length=max_seq_length, pad_val=tokenizer.pad_token_id)
        token_type_ids = seq_padding(o['token_type_ids'], max_seq_length=max_seq_length, pad_val=0)

        d = {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'labels': input_ids,
            'seqlen': seqlen
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

        max_len = torch.max(o.pop('seqlen'))
        o['input_ids'] = o['input_ids'][:, :max_len]
        o['token_type_ids'] = o['token_type_ids'][:, :max_len]
        o['labels'] = o['labels'][:, :max_len]
        return o


# 教师12层
class TeacherTransformer(TransformerModelForUnilm, with_pl=True):
    def __init__(self, *args, **kwargs):
        super(TeacherTransformer, self).__init__(*args, **kwargs)

    def compute_loss(self, *args, **batch) -> tuple:
        batch['attention_mask'] = unilm_mask(batch['token_type_ids'])
        if getattr(self.config, 'type_vocab_size', 0) != 2:
            batch.pop('token_type_ids')

        labels = batch.pop('labels', None)
        outputs = self.model(*args, **batch)
        hidden_states = outputs[0]
        lm_logits = self.model.lm_head(hidden_states)

        if labels is not None:
            labels = labels.long()
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.model.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            outputs = (loss, lm_logits, labels)
        else:
            outputs = (lm_logits,)
        return outputs


# 学生6层
class StudentTransformer(TransformerModelForUnilm, with_pl=True):
    def __init__(self, teacher_model, *args, **kwargs):
        super(StudentTransformer, self).__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.kl_loss = KLDivLoss('sum')

    def compute_loss(self, *args, **batch) -> tuple:
        labels = batch.pop('labels', None)

        inputs = {k: v for k, v in batch.items()}
        inputs['attention_mask'] = unilm_mask(inputs['token_type_ids'])
        if getattr(self.config, 'type_vocab_size', 0) != 2:
            inputs.pop('token_type_ids')

        outputs = self.model(*args, **inputs, output_hidden_states=True)
        # hidden_states = outputs[0]
        # 第六层
        hidden_states = outputs[2][-6]
        lm_logits = self.model.lm_head(hidden_states)
        if labels is not None:
            labels = labels.long()
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_student = self.model.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            teacher_logits = self.teacher_model.compute_loss(*args, **batch)[0]
            kl_Loss = self.kl_loss([teacher_logits, lm_logits])
            loss_dict = {
                'loss_student': loss_student,
                'kl_Loss': kl_Loss,
                'loss': loss_student * 0.1 + kl_Loss
            }

            outputs = (loss_dict, lm_logits, labels)
        else:
            outputs = (lm_logits,)
        return outputs


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
    model_args, training_args, data_args = parser.parse_dict(train_info_args)

    checkpoint_callback = SimpleModelCheckpoint(monitor="loss",
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
        strategy='ddp' if torch.cuda.device_count() > 1 else 'auto',
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


    # 是否首先训练模型
    is_training_teacher = True

    if is_training_teacher:  # 训练teacher 模型
        model = TeacherTransformer(config=config, model_args=model_args, training_args=training_args)
    else:  # 蒸馏模型
        teacher_weight = './best_teacher.pt'
        # 加载训练好的权重
        teacher_model = TeacherTransformer.load_from_checkpoint(teacher_weight, config=config, model_args=model_args,
                                                                training_args=training_args)
        for k, p in teacher_model.named_parameters():
            p.requires_grad = False
        model = StudentTransformer(teacher_model, config=config, model_args=model_args, training_args=training_args)

    if not data_args.convert_onnx:
        train_datasets = dataHelper.load_distributed_random_sampler(
            dataHelper.train_files,
            with_load_memory=True,
            collate_fn=dataHelper.collate_fn,
            batch_size=training_args.train_batch_size,
            num_processes = trainer.world_size, process_index=trainer.global_rank)
        if train_datasets is not None:
            trainer.fit(model, train_dataloaders=train_datasets)
        else:
            eval_datasets = dataHelper.load_sequential_sampler(dataHelper.eval_files,batch_size=training_args.eval_batch_size,collate_fn=dataHelper.collate_fn)
            test_datasets = dataHelper.load_sequential_sampler(dataHelper.test_files,batch_size=training_args.test_batch_size,collate_fn=dataHelper.collate_fn)
            if eval_datasets is not None:
                trainer.validate(model, dataloaders=eval_datasets, ckpt_path='./best.pt')

            if test_datasets is not None:
                trainer.test(model, dataloaders=test_datasets, ckpt_path='best.pt')

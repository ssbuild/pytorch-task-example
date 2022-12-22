# -*- coding: utf-8 -*-
import json
import typing

import numpy as np
import torch
from deep_training.data_helper import DataHelper
from deep_training.data_helper import ModelArguments, DataArguments, TrainingArguments
from deep_training.data_helper import load_tokenizer_and_config_with_args
from deep_training.nlp.layers.mask import unilm_mask
from deep_training.nlp.losses.loss_kl import KLDivLoss
from deep_training.nlp.models.transformer import TransformerModelForUnilm
from deep_training.utils.func import seq_padding
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, IterableDataset
from transformers import BertTokenizer
from transformers import HfArgumentParser

train_info_args = {
    'devices':  1,
    'data_backend': 'memory_raw',
    'model_type': 'bert',
    'model_name_or_path':'/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'tokenizer_name':'/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'config_name':'/data/nlp/pre_models/torch/bert/bert-base-chinese/config.json',
    'do_train': True,
    'train_file':'/data/nlp/nlp_train_data/thucnews/train.json',
    'max_steps': 100000,
    'train_batch_size':8,
    'test_batch_size':2,
    'adam_epsilon':1e-8,
    'gradient_accumulation_steps':1,
    'max_grad_norm':1.0,
    'weight_decay':0,
    'warmup_steps':0,
    'output_dir':'./output',
    'max_seq_length':200,
    'max_target_length':50
}

class NN_DataHelper(DataHelper):
    # 切分词
    def on_data_process(self, data: typing.Any, user_data: tuple):
        tokenizer: BertTokenizer
        tokenizer, max_seq_length, do_lower_case, label2id, mode = user_data
        x = data
        assert isinstance(x,tuple)
        o = tokenizer.encode_plus(text=x[0], text_pair=x[1], max_length=max_seq_length, truncation=True)
        seqlen = np.asarray(len(o['input_ids']),dtype=np.int32)
        input_ids = seq_padding(o['input_ids'],max_seq_length=max_seq_length,pad_val=tokenizer.pad_token_id)
        token_type_ids = seq_padding(o['token_type_ids'],max_seq_length=max_seq_length,pad_val=0)

        d = {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'labels': input_ids,
            'seqlen': seqlen
        }
        return d


    # 读取文件
    def on_get_corpus(self, files:typing.List, mode:str):
        D = []
        for filename in files:
            with open(filename, mode='r', encoding='utf-8') as f:
                lines = f.readlines()
                for i,line in enumerate(lines):
                    jd = json.loads(line)
                    D.append((jd['content'], jd['title']))
                    if i > 1000:
                        break
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
        o['token_type_ids'] = o['token_type_ids'][:, :max_len]
        o['labels'] = o['labels'][:, :max_len]
        return o

class TeacherTransformer(TransformerModelForUnilm, with_pl=True):
    def __init__(self, *args,**kwargs):
        super(TeacherTransformer, self).__init__(*args,**kwargs)

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
            loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            outputs = (loss, lm_logits, labels)
        else:
            outputs = (lm_logits,)
        return outputs

class StudentTransformer(TransformerModelForUnilm, with_pl=True):
    def __init__(self, *args,**kwargs):
        super(StudentTransformer, self).__init__(*args,**kwargs)
        self.teacher_model = TeacherTransformer(*args,**kwargs)
        for k,p in self.teacher_model.named_parameters():
            p.requires_grad=False
        self.kl_loss = KLDivLoss('sum')

    def compute_loss(self, *args,**batch) -> tuple:
        labels = batch.pop('labels', None)

        inputs = {k:v for k,v in batch.items()}
        inputs['attention_mask'] = unilm_mask(inputs['token_type_ids'])
        if getattr(self.config, 'type_vocab_size', 0) != 2:
            inputs.pop('token_type_ids')


        outputs = self.model(*args,**inputs)
        hidden_states = outputs[0]
        lm_logits = self.model.lm_head(hidden_states)
        if labels is not None:
            labels = labels.long()
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_student = self.model.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            teacher_logits = self.teacher_model.compute_loss(*args,**batch)[0]
            kl_Loss = self.kl_loss([teacher_logits,lm_logits])
            loss_dict = {
                'loss_student': loss_student,
                'kl_Loss': kl_Loss,
                'loss': loss_student * 0.1 + kl_Loss
            }

            outputs = (loss_dict,lm_logits,labels)
        else:
            outputs = (lm_logits,)
        return outputs

    # def validation_epoch_end(self, outputs: typing.Union[EPOCH_OUTPUT, typing.List[EPOCH_OUTPUT]]) -> None:
    #     self.index += 1
    #     # if self.index < 2:
    #     #     self.log('val_f1', 0.0, prog_bar=True)
    #     #     return
    #
    #     y_preds, y_trues = [], []
    #     for i, o in tqdm(enumerate(outputs), total=len(outputs)):
    #         logits, _ = o['outputs']
    #         bs = len(logits)
    #         output_labels = eval_labels[i * bs:(i + 1) * bs]
    #         p_spoes = extract_spoes(logits, self.config.id2label, self.rel2id, threshold)
    #         t_spoes = output_labels
    #         y_preds.extend(p_spoes)
    #         y_trues.extend(t_spoes)
    #
    #     print(y_preds[:3])
    #     print(y_trues[:3])
    #     f1, str_report = metric_for_spo(y_trues, y_preds, self.rel2id)
    #     print(f1)
    #     print(str_report)
    #     self.log('val_f1', f1, prog_bar=True)
    #
    # def test_epoch_end(self, outputs: typing.Union[EPOCH_OUTPUT, typing.List[EPOCH_OUTPUT]]) -> None:
    #     print('write to file')
    #     from fastdatasets.record import NumpyWriter
    #     f = NumpyWriter('./eval_vecs.record')
    #
    #     for i, o in tqdm(enumerate(outputs), total=len(outputs)):
    #         _, b_logits, b_labels = o['outputs']
    #         for j in range(len(b_logits)):
    #             obj = {
    #                 'logit': np.asarray(b_logits[j], dtype=np.float32),
    #                 'label': np.asarray(b_labels[j], dtype=np.int32),
    #             }
    #             f.write(obj)
    #     f.close()



if __name__== '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
    model_args, training_args, data_args = parser.parse_dict(train_info_args)

    checkpoint_callback = ModelCheckpoint(monitor="loss", every_n_train_steps=1000)
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
    
    model = StudentTransformer(config=config,model_args=model_args,training_args=training_args)

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
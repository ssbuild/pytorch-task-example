# -*- coding: utf-8 -*-
# @Time    : 2023/2/10 17:18

import logging

import Levenshtein
import numpy as np
import torch
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from deep_training.nlp.metrics.pointer import metric_for_pointer
from deep_training.nlp.models.transformer import TransformerForSeq2SeqLM
from deep_training.utils.trainer import SimpleModelCheckpoint
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import HfArgumentParser, T5ForConditionalGeneration

from data_utils import train_info_args, NN_DataHelper


class MyTransformer(TransformerForSeq2SeqLM, with_pl=True):
    def __init__(self, *args, **kwargs):
        super(MyTransformer, self).__init__(*args, **kwargs)



class MySimpleModelCheckpoint(SimpleModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super(MySimpleModelCheckpoint, self).__init__(*args, **kwargs)
        self.weight_file = './best.pt'

    @staticmethod
    def generate_text_huggingface(pl_module: MyTransformer, input_ids, tokenizer, max_target_length, device=0):
        device = torch.device('cuda:{}'.format(device))

        input_ids = torch.tensor(input_ids, dtype=torch.int32,device = device).unsqueeze(0)
        output = pl_module.backbone.model.generate(input_ids,
                   max_length = max_target_length,
                   bos_token_id = tokenizer.cls_token_id,
                   pad_token_id = tokenizer.pad_token_id,
                   eos_token_id = tokenizer.sep_token_id,
        )

        gen_tokens = []
        gen_ids =  output[0].cpu().numpy()
        for logits in output[0]:
            # gen_ids.append(logits.cpu().numpy())
            token = tokenizer._convert_id_to_token(logits)
            if token.startswith('##'):
                token = token.replace('##', '')
            gen_tokens.append(token)
        return ''.join(gen_tokens),gen_ids

    def on_save_model(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        pl_module: MyTransformer

        # 当前设备
        device = torch.device('cuda:{}'.format(trainer.global_rank))
        eval_datasets = dataHelper.load_sequential_sampler(dataHelper.eval_files,
                                                           batch_size=training_args.eval_batch_size,
                                                           collate_fn=dataHelper.collate_fn)

        config = pl_module.config

        y_preds, y_trues = [], []

        op_map = {
            'insert': 0,
            'delete': 1,
            'replace': 2
        }

        #  三元组（action,position,vocab）
        def get_ops(source,target):
            edits = Levenshtein.opcodes(source, target)
            ops = []
            for item in edits:
                if item[0] == 'equal':
                    continue
                action = op_map[item[0]]
                s = item[1]
                e = item[2]
                ds = item[3]
                de = item[4]
                #insert,replace
                if action == 0 or action == 2:
                    for idx in range(de-ds):
                        ops.append((action, s+idx, target[ds + idx]))
                #delete
                elif action == 1:
                    for idx in range(s, e):
                        ops.append((action, s+idx, 0))
                else:
                    raise ValueError('invalid action ',action)

            return ops

        for i, batch in tqdm(enumerate(eval_datasets), total=len(eval_datasets), desc='evalute'):
            batch_labels = batch.pop('labels',None)
            for k in batch:
                batch[k] = batch[k].to(device)
            for input_ids,attention_mask,labels in zip(batch['input_ids'],batch['attention_mask'],batch_labels):
                seqlen = torch.sum(attention_mask,dim=-1)
                output = MySimpleModelCheckpoint.generate_text_huggingface(pl_module,
                                                                            input_ids,
                                                                            tokenizer=tokenizer,
                                                                            max_target_length=data_args.max_target_length,
                                                                            device=trainer.global_rank)
                source = input_ids[1:seqlen-1].cpu().numpy()
                #  三元组（action,position,vocab）
                pred_ops = get_ops(source, output[1])


                _ = np.where(labels==-100)[0]
                if len(_):
                    seqlen = _[0] + 1
                else:
                    seqlen = len(labels)
                labels = labels[1:seqlen - 1]
                #  三元组（action,position,vocab）
                true_ops = get_ops(source, labels)

                y_preds.append(pred_ops)
                y_trues.append(true_ops)

        print(y_preds[:3])
        print(y_trues[:3])

        label2id = {
            'insert': 0,
            'delete': 1,
            'replace': 2
        }

        f1, str_report = metric_for_pointer(y_trues, y_preds, label2id)
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

    checkpoint_callback = MySimpleModelCheckpoint(every_n_epochs=1,
                                                  every_n_train_steps=2000)
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


    model = MyTransformer(config=config, model_args=model_args, training_args=training_args)

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
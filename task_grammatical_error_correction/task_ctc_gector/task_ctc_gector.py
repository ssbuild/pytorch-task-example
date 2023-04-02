# -*- coding: utf-8 -*-
# @Time    : 2023/2/10 17:18

import logging
import numpy as np
import torch
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from deep_training.nlp.metrics.pointer import metric_for_pointer
from deep_training.nlp.models.gec_model import TransformerForGec, extract_gec, extract_gec_from_labels
from deep_training.utils.trainer import SimpleModelCheckpoint
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import HfArgumentParser
from data_utils import NN_DataHelper, train_info_args


class MyTransformer(TransformerForGec, with_pl=True):
    def __init__(self, *args, **kwargs):
        super(MyTransformer, self).__init__(*args, **kwargs)



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
        eval_datasets = dataHelper.load_sequential_sampler(dataHelper.eval_files,batch_size=training_args.eval_batch_size,collate_fn=dataHelper.collate_fn)

        config = pl_module.config

        y_preds, y_trues = [], []
        for i, batch in tqdm(enumerate(eval_datasets), total=len(eval_datasets), desc='evalute'):
            for k in batch:
                batch[k] = batch[k].to(device)
            o = pl_module.validation_step(batch, i)
            logits_action,logits_probs,seqlens,labels_action,labels_probs = o['outputs']
            #抽取 三元组（action,position,vocab）
            output_list = extract_gec([logits_action,logits_probs,seqlens])
            true_list = extract_gec_from_labels([labels_action,labels_probs,seqlens])
            # y_preds.extend(output_list)
            # y_trues.extend(true_list)
            for ones in output_list:
                y_preds.append( [(_[0]-1,*_[1:]) for _ in ones])

            for ones in true_list:
                y_trues.append([(_[0] - 1, *_[1:]) for _ in ones])


        #  三元组（action,position,vocab）
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
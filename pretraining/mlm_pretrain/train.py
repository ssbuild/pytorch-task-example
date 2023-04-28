# -*- coding: utf-8 -*-
import os
import random

import torch
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments, MlmDataArguments
from deep_training.nlp.models.transformer import TransformerForMaskLM
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, IterableDataset
from transformers import HfArgumentParser

from data_utils import NN_DataHelper, train_info_args
from torch.nn.functional import one_hot

mask_token_id = None


class MyTransformer(TransformerForMaskLM, with_pl=True):
    def __init__(self, *args, **kwargs):
        super(MyTransformer, self).__init__(*args, **kwargs)
        self.loss_fct = CrossEntropyLoss(reduction='none')

    def compute_loss_mlm(self, y_trues, y_preds, mask):
        y_preds = torch.transpose(y_preds, 1, 2)
        masked_lm_loss = self.loss_fct(y_preds, y_trues)
        masked_lm_loss = torch.sum(mask * masked_lm_loss) / (torch.sum(mask) + 1e-8)
        return masked_lm_loss

    def compute_acc(self, y_trues, y_preds, mask):
        acc = torch.eq(torch.argmax(y_preds, dim=-1), y_trues)
        acc = torch.sum(mask * acc) / (torch.sum(mask) + 1e-8)
        return acc

    def compute_loss(self, *args, **batch) -> tuple:
        labels = None
        mask = None
        if 'labels' in batch:
            labels = batch.pop('labels')
            mask = batch.pop('mask')

        outputs = self.model(*args, **batch)
        logits = outputs[0]
        if labels is not None:
            loss = self.compute_loss_mlm(labels, logits, mask)
            acc = self.compute_acc(labels, logits, batch['attention_mask'])
            mlm_acc = self.compute_acc(labels, logits, mask)
            loss = {
                'loss': loss,
                'acc': acc,
                'mlm_acc': mlm_acc,
            }
            outputs = (loss, logits, labels)
        else:
            outputs = (logits,)
        return outputs


if __name__ == '__main__':

    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, MlmDataArguments))
    model_args, training_args, data_args, mlm_data_args = parser.parse_dict(train_info_args)

    checkpoint_callback = ModelCheckpoint(save_last=True,
                                          verbose=True,
                                          monitor="loss",
                                          save_top_k=5,
                                          every_n_train_steps=2000 // training_args.gradient_accumulation_steps)
    trainer = Trainer(
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='step')],
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

    rng = random.Random(training_args.seed)
    dataHelper = NN_DataHelper(model_args, training_args, data_args, mlm_args=(
    rng, mlm_data_args.do_whole_word_mask, mlm_data_args.max_predictions_per_seq, mlm_data_args.masked_lm_prob))
    tokenizer, config, label2id, id2label = dataHelper.load_tokenizer_and_config()
    mask_token_id = tokenizer.mask_token_id
    # 缓存数据集
    if data_args.do_train:
        dataHelper.make_dataset_with_args(data_args.train_file, mixed_data=False, shuffle=True, mode='train',
                                          dupe_factor=mlm_data_args.dupe_factor, num_process_worker=10)
    if data_args.do_eval:
        dataHelper.make_dataset_with_args(data_args.eval_file, shuffle=False, mode='eval')
    if data_args.do_test:
        dataHelper.make_dataset_with_args(data_args.test_file, mode='test')

    model = MyTransformer(config=config, model_args=model_args, training_args=training_args)

    if not data_args.convert_onnx:
        train_datasets = dataHelper.load_random_sampler(dataHelper.train_files,
                                                        with_load_memory=False,
                                                        with_record_iterable_dataset=True,
                                                        collate_fn=dataHelper.collate_fn,
                                                        batch_size=training_args.train_batch_size,
                                                        shuffle=True, infinite=True, num_processes=trainer.world_size,
                                                        process_index=trainer.global_rank)
        # 恢复断点训练
        resume_ckpt_path = r'./epoch=0-step=4200.ckpt'
        if not os.path.exists(resume_ckpt_path):
            resume_ckpt_path = None

        if train_datasets is not None:
            trainer.fit(model, train_dataloaders=train_datasets, ckpt_path=resume_ckpt_path)
        else:
            eval_datasets = dataHelper.load_sequential_sampler(dataHelper.eval_files,
                                                               batch_size=training_args.eval_batch_size,
                                                               collate_fn=dataHelper.collate_fn)
            test_datasets = dataHelper.load_sequential_sampler(dataHelper.test_files,
                                                               batch_size=training_args.test_batch_size,
                                                               collate_fn=dataHelper.collate_fn)
            if eval_datasets is not None:
                trainer.validate(model, dataloaders=eval_datasets, ckpt_path='./best.pt')

            if test_datasets is not None:
                trainer.test(model, dataloaders=test_datasets, ckpt_path='best.pt')

# -*- coding: utf-8 -*-
import random

import torch
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments, MlmDataArguments
from deep_training.nlp.models.transformer import TransformerForMaskLM
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, IterableDataset
from transformers import HfArgumentParser

from data_utils import NN_DataHelper,train_info_args

class MyTransformer(TransformerForMaskLM, with_pl=True):
    def __init__(self, *args, **kwargs):
        super(MyTransformer, self).__init__(*args, **kwargs)
        self.loss_fct = CrossEntropyLoss(reduction='none', ignore_index=self.config.pad_token_id)

    def compute_loss_mlm(self, y_trues, y_preds, weight):
        y_preds = torch.transpose(y_preds, 1, 2)
        loss = self.loss_fct(y_preds, y_trues)
        loss = torch.sum(loss * weight, dtype=torch.float) / (torch.sum(weight, dtype=torch.float) + 1e-12)
        return loss.mean()

    def compute_loss(self, *args, **batch) -> tuple:
        labels, weight = None, None
        if 'labels' in batch:
            weight = batch.pop('weight')
            labels = batch.pop('labels')
        outputs = self.model(*args, **batch)
        logits = outputs[0]
        if labels is not None:
            loss = self.compute_loss_mlm(labels, logits, weight)
            outputs = (loss, logits, labels)
        else:
            outputs = (logits,)
        return outputs


if __name__ == '__main__':

    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, MlmDataArguments))
    model_args, training_args, data_args, mlm_data_args = parser.parse_dict(train_info_args)

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

    rng = random.Random(training_args.seed)
    dataHelper = NN_DataHelper(model_args, training_args, data_args,mlm_args = (rng, mlm_data_args.do_whole_word_mask, mlm_data_args.max_predictions_per_seq,mlm_data_args.masked_lm_prob))
    tokenizer, config, label2id, id2label = dataHelper.load_tokenizer_and_config()

    # 缓存数据集
    if data_args.do_train:
        dataHelper.make_dataset_with_args(data_args.train_file,shuffle=True,mode='train', dupe_factor=mlm_data_args.dupe_factor)
    if data_args.do_eval:
        dataHelper.make_dataset_with_args(data_args.eval_file,shuffle=False,mode='eval')
    if data_args.do_test:
        dataHelper.make_dataset_with_args(data_args.test_file,shuffle=False,mode='test')

    model = MyTransformer(config=config, model_args=model_args, training_args=training_args)

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

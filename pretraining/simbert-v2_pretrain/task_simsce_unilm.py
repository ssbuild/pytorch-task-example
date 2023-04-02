# -*- coding: utf-8 -*-

import torch
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from deep_training.nlp.layers.mask import unilm_mask
from deep_training.nlp.losses.contrast import SimcseLoss
from deep_training.nlp.models.transformer import TransformerModelForUnilm
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader, IterableDataset
from transformers import HfArgumentParser
from data_utils import NN_DataHelper,train_info_args


class MyTransformer(TransformerModelForUnilm, with_pl=True):
    def __init__(self, *args, **kwargs):
        super(MyTransformer, self).__init__(*args, **kwargs)
        config = self.config
        self.sim_head = nn.Linear(config.hidden_size, 512, bias=False)
        self.loss_fn = SimcseLoss()

    def get_model_lr(self):
        return super(MyTransformer, self).get_model_lr() + [
            (self.sim_head, self.config.task_specific_params['learning_rate_for_task'])
        ]

    def compute_loss(self, *args, **batch) -> tuple:
        if self.training:
            batch = {k: torch.repeat_interleave(v, 2, dim=0) for k, v in batch.items()}
        labels = batch.pop('labels', None)
        batch['attention_mask'] = unilm_mask(batch['token_type_ids'])
        outputs = self.model(*args, **batch)
        lm_logits = self.model.lm_head(outputs[0])
        simcse_logits = self.sim_head(outputs[1])

        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss1 = self.model.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss2 = self.loss_fn(simcse_logits)
            loss = loss1 + loss2
            loss_dict = {
                'loss': loss,
                'unilm_loss': loss1,
                'simcse_loss': loss2,
            }
            outputs = (loss_dict, lm_logits, simcse_logits)
            self.log_dict(loss_dict, prog_bar=True)
        else:
            outputs = (lm_logits, simcse_logits)
        return outputs


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
    model_args, training_args, data_args = parser.parse_dict(train_info_args)

    checkpoint_callback = ModelCheckpoint(monitor="loss",
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
        dataHelper.make_dataset_with_args(data_args.train_file,mixed_data=False, shuffle=True,mode='train')
    if data_args.do_eval:
        dataHelper.make_dataset_with_args(data_args.eval_file, mode='eval')
    if data_args.do_test:
        dataHelper.make_dataset_with_args(data_args.test_file,mode='test')

    model = MyTransformer(config=config, model_args=model_args, training_args=training_args)

    if not data_args.convert_onnx:
        train_datasets = dataHelper.load_random_sampler(dataHelper.train_files,
                                                        with_load_memory=False,
                                                        with_record_iterable_dataset=True,
                                                        collate_fn=dataHelper.collate_fn,
                                                        batch_size=training_args.train_batch_size,
                                                        shuffle=True, infinite=True, num_processes=trainer.world_size,
                                                        process_index=trainer.global_rank)
        if train_datasets is not None:
            trainer.fit(model, train_dataloaders=train_datasets)
        else:
            eval_datasets = dataHelper.load_sequential_sampler(dataHelper.eval_files,batch_size=training_args.eval_batch_size,collate_fn=dataHelper.collate_fn)
            test_datasets = dataHelper.load_sequential_sampler(dataHelper.test_files,batch_size=training_args.test_batch_size,collate_fn=dataHelper.collate_fn)
            if eval_datasets is not None:
                trainer.validate(model, dataloaders=eval_datasets, ckpt_path='./best.pt')

            if test_datasets is not None:
                trainer.test(model, dataloaders=test_datasets, ckpt_path='best.pt')

# -*- coding: utf-8 -*-
import numpy as np
import torch
from deep_training.data_helper import ModelArguments, DataArguments, TrainingArguments
from deep_training.data_helper import load_tokenizer_and_config_with_args
from deep_training.nlp.models.transformer import TransformerModelForUnilm
from deep_training.utils.trainer import SimpleModelCheckpoint
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, IterableDataset
from transformers import HfArgumentParser, BertTokenizer

from data_utils import NN_DataHelper, data_conf

train_info_args = {
    'devices': 1,
    'data_backend': 'record',
    'model_type': 'bert',
    'model_name_or_path':'/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'tokenizer_name':'/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'config_name':'/data/nlp/pre_models/torch/bert/bert-base-chinese/config.json',
    # 语料已经制作好，不需要在转换
    'convert_file': False,
    'do_train': True,
    'train_file':'./output/dataset_0-train.record',
    'max_epochs': 10,
    'train_batch_size':8,
    'eval_batch_size': 2,
    'test_batch_size':2,
    'learning_rate': 5e-5,
    'adam_epsilon':1e-8,
    'gradient_accumulation_steps':1,
    'max_grad_norm':1.0,
    'weight_decay':0,
    'warmup_steps':0,
    'output_dir':'./output',
    'max_seq_length': 512,
    'max_target_length': 50 #预测最大长度
}


class MyTransformer(TransformerModelForUnilm, with_pl=True):
    def __init__(self, *args,**kwargs):
        super(MyTransformer, self).__init__(*args,**kwargs)



class MySimpleModelCheckpoint(SimpleModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super(MySimpleModelCheckpoint, self).__init__(*args, **kwargs)
        self.weight_file = './best.pt'


    def generate_text(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",prefix):
        pl_module: MyTransformer

        # 当前设备
        device = torch.device('cuda:{}'.format(trainer.global_rank))

        self.tokenizer: BertTokenizer
        tokenizer = self.tokenizer
        data_args = self.data_args

        # 简易测试生成
        o = tokenizer.encode_plus(prefix, truncation=True, max_length=512)
        gen_ids, gen_tokens = [], []
        batch = {}
        for i in range(data_args.max_target_length):
            batch.clear()

            if len(gen_ids) == 0:
                batch['input_ids'] = [o['input_ids'] ]
                batch['token_type_ids'] = [o['token_type_ids']]
            else:
                batch['input_ids'] = [o['input_ids'] + gen_ids + [tokenizer.sep_token_id]]
                batch['token_type_ids'] = [o['token_type_ids'] + [1] * (len(gen_ids) + 1)]

            for k in batch:
                batch[k] = torch.tensor(batch[k], dtype=torch.int32)
            for k in batch:
                batch[k] = batch[k].to(device)

            out = pl_module.test_step(batch, 0)
            logits = out['outputs'][0]
            logits = np.argmax(logits[:, -1], axis=-1)
            logits = logits[0]
            gen_ids.append(logits)
            token = tokenizer._convert_id_to_token(logits)
            if token.startswith('##'):
                token = token.replace('##','')
            gen_tokens.append(token)

        print('input', prefix)

        for k in batch:
            batch[k] = batch[k].cpu()

        print(batch)
        print('output', ''.join(gen_tokens))


    def on_save_model(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        #保存权重
        super(MySimpleModelCheckpoint, self).on_save_model(trainer,pl_module)
        special = data_conf['special']
        prefixs = [('七律','归山吟寄友'),
                   ('五绝','钓鱼有感')
        ]
        for prefix in prefixs:
            prefix = special[prefix[0]] + prefix[1]
            self.generate_text(trainer,pl_module,prefix)


if __name__== '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
    model_args, training_args, data_args = parser.parse_dict(train_info_args)
    # 保存最小loss模型
    checkpoint_callback = MySimpleModelCheckpoint(monitor="loss", every_n_train_steps=2000 // training_args.gradient_accumulation_steps)
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

    #赋值
    checkpoint_callback.tokenizer = tokenizer
    checkpoint_callback.data_args = data_args


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
                                             with_load_memory=True,
                                             with_record_iterable_dataset=False,)
    
    if train_datasets is not None:
        train_datasets = DataLoader(train_datasets, batch_size=training_args.train_batch_size,
                                    collate_fn=dataHelper.collate_fn,
                                    shuffle=False if isinstance(train_datasets, IterableDataset) else True)
    
    model = MyTransformer(config=config,model_args=model_args,training_args=training_args)

    if train_datasets is not None:
        trainer.fit(model, train_dataloaders=train_datasets)
    else:
        #加载权重
        model = MyTransformer.load_from_checkpoint('./best.pt', config=config,
                                                   model_args=model_args,
                                                   training_args=training_args)

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

        is_convert_onnx = True
        # 是否转换模型
        if is_convert_onnx:
            input_sample = (
                torch.ones(size=(1, 128), dtype=torch.int64),
                torch.ones(size=(1, 128), dtype=torch.int64),
            )
            model.eval()
            model.to('cuda')
            input_names = ["input_ids", "token_type_ids"]
            out_names = ["pred_ids"]


            model.to_onnx('./best.onnx',
                          input_sample=input_sample,
                          verbose=True,
                          opset_version=14,
                          do_constant_folding=True,
                          input_names=input_names,
                          output_names=out_names,
                          dynamic_axes={"input_ids": [0, 1],
                                        "attention_mask": [0, 1],
                                        "pred_ids": [0, 1]
                                        }
                          )
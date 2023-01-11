# -*- coding: utf-8 -*-
import logging
import os.path
import typing

import numpy as np
import scipy
import torch
from deep_training.data_helper import DataHelper
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from deep_training.data_helper import load_tokenizer_and_config_with_args
from deep_training.nlp.losses.focal_loss import FocalLoss
from deep_training.nlp.losses.loss_arcface import ArcMarginProduct
from deep_training.nlp.models.transformer import TransformerModel
from deep_training.utils.trainer import SimpleModelCheckpoint
from pytorch_lightning import Trainer
from torch import nn
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import HfArgumentParser, BertTokenizer

model_base_dir = '/data/torch/bert-base-chinese'
#model_base_dir = '/data/nlp/pre_models/torch/bert/bert-base-chinese'

train_info_args = {
    'devices': torch.cuda.device_count(),
    'data_backend': 'record',
    'model_type': 'bert',
    'model_name_or_path': model_base_dir,
    'tokenizer_name': model_base_dir,
    'config_name': os.path.join(model_base_dir, 'config.json'),
    # 语料已经制作好，不需要在转换
    'convert_file': False,
    'do_train': True,
    'do_eval': True,
    'do_test': False,
    'train_file': '/data/record/cse_0110/train.record',
    'eval_file': '/data/record/cse_0110/eval.record',
    # 'test_file': '/home/tk/train/make_big_data/output/eval.record',
    'label_file': '/data/record/cse_0110/labels_122.txt',
    'learning_rate': 3e-5,
    'max_steps': 500000,
    'max_epochs': 1,
    'train_batch_size': 10,
    'eval_batch_size': 10,
    'test_batch_size': 10,
    'adam_epsilon': 1e-8,
    'gradient_accumulation_steps': 2,
    'max_grad_norm': 1.0,
    'weight_decay': 0,
    'warmup_steps': 0,
    'output_dir': './output',
    'train_max_seq_length': 512,
    'eval_max_seq_length': 512,
    'test_max_seq_length': 512,
}


class NN_DataHelper(DataHelper):
    # 切分词
    def on_data_process(self, data: typing.Any, user_data: tuple):
        tokenizer: BertTokenizer
        tokenizer, max_seq_length, do_lower_case, label2id, mode = user_data
        sentence, label_str = data

        o = tokenizer(sentence, max_length=max_seq_length, truncation=True, add_special_tokens=True, )
        input_ids = np.asarray(o['input_ids'], dtype=np.int64)
        attention_mask = np.asarray(o['attention_mask'], dtype=np.int64)

        labels = np.asarray(label2id[label_str] if label_str is not None else 0, dtype=np.int64)
        seqlen = np.asarray(len(input_ids), dtype=np.int64)
        pad_len = max_seq_length - len(input_ids)
        if pad_len > 0:
            pad_val = tokenizer.pad_token_id
            input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            attention_mask = np.pad(attention_mask, (0, pad_len), 'constant', constant_values=(0, 0))
        d = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'seqlen': seqlen
        }
        return d

    # 读取标签
    def on_get_labels(self, files: typing.List[str]):
        file = files[0]
        with open(file, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
        labels = []
        for line in lines:
            line = line.replace('\r\n', '').replace('\n', '')
            if not line:
                continue
            labels.append(line)
        labels = list(set(labels))
        labels = sorted(labels)
        label2id = {l: i for i, l in enumerate(labels)}
        id2label = {i: l for i, l in enumerate(labels)}
        self.label2id = label2id
        self.id2label = id2label
        return self.label2id, self.id2label

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

        o.pop('id', None)
        max_len = torch.max(o.pop('seqlen'))

        o['input_ids'] = o['input_ids'][:, :max_len]
        o['attention_mask'] = o['attention_mask'][:, :max_len]
        if 'token_type_ids' in o:
            o['token_type_ids'] = o['token_type_ids'][:, :max_len]

        return o
    
def transform_and_normalize(vecs, kernel=None, bias=None):
    """应用变换，然后标准化
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    norms = (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5
    return vecs / np.clip(norms, 1e-8, np.inf)


def compute_corrcoef(x, y):
    """Spearman相关系数
    """
    return scipy.stats.spearmanr(x, y).correlation

def generate_pair_example(all_example_dict: dict):
    all_example_dict = copy.copy(all_example_dict)

    all_example_pos,all_example_neg = [],[]
    all_keys = list(all_example_dict.keys())
    np.random.shuffle(all_keys)

    for pos_label in all_keys:
        examples = all_example_dict[pos_label]
        if len(examples) == 0:
            continue
        num_size = int(min(np.random.randint(300,1000), int(len(examples) * 0.5)))
        if num_size < 2:
            continue
        id_list = list(range(len(examples)))
        ids = np.random.choice(id_list, replace=False, size=num_size)
        ids = sorted(ids,reverse=True)
        for i1,i2 in zip(ids[::2],ids[1::2]):
            v1 = examples[i1]
            v2 = examples[i2]
            examples.pop(i1)
            examples.pop(i2)
            all_example_pos.append((v1, v2))
        # 去除空标签数据
        if len(examples) <= 1:
            all_keys.remove(pos_label)

    flat_examples = []
    for k in all_keys:
        d_list = all_example_dict[k]
        for d in d_list:
            flat_examples.append((k,d))
    print('construct neg from {} flat_examples'.format(len(flat_examples)))
    idx_list = list(range(len(flat_examples)))
    np.random.shuffle(idx_list)
    while len(idx_list) >= 2:
        flag = False
        k1,e1 = flat_examples[idx_list.pop(0)]
        for i in idx_list[1:]:
            k2,e2 = flat_examples[i]
            if k1 != k2:
                all_example_neg.append((e1,e2))
                idx_list.remove(i)
                if len(all_example_neg) > len(all_example_pos) * 10:
                    flag = True
                break
        if flag:
            break
    print('pos num',len(all_example_pos),'neg num',len(all_example_neg) )
    return all_example_pos,all_example_neg


def evaluate_sample(a_vecs,b_vecs,labels):
    a_vecs = transform_and_normalize(a_vecs)
    b_vecs = transform_and_normalize(b_vecs)
    sims = (a_vecs * b_vecs).sum(axis=1)
    corrcoef = compute_corrcoef(labels,sims)
    print('*' * 30)
    print('spearman ', corrcoef)
    return corrcoef

class MyTransformer(TransformerModel, with_pl=True):
    def __init__(self,*args,**kwargs):
        super(MyTransformer, self).__init__(*args,**kwargs)
        self.feat_head = nn.Linear(self.config.hidden_size, 512, bias=False)
        self.metric_product = ArcMarginProduct(512,self.config.num_labels,s=30.0, m=0.50, easy_margin=False)

        loss_type = 'focal_loss'
        if loss_type == 'focal_loss':
            self.loss_fn = FocalLoss(gamma=2)
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss()

    def get_model_lr(self):
        return super(MyTransformer, self).get_model_lr() + [
            (self.feat_head, self.config.task_specific_params['learning_rate_for_task']),
            (self.metric_product, self.config.task_specific_params['learning_rate_for_task']),
            (self.loss_fn, self.config.task_specific_params['learning_rate_for_task'])
        ]

    def compute_loss(self, *args,**batch) -> tuple:
        labels: torch.Tensor = batch.pop('labels',None)
        outputs = self.model(*args,**batch)
        logits = self.feat_head(outputs[0][:, 0, :])
        # logits = torch.tan(logits)
        # logits = F.normalize(logits)
        if labels is not None:
            labels = torch.squeeze(labels, dim=1)
            metric_logits = self.metric_product(logits, labels)
            loss = self.loss_fn(metric_logits, labels)
            outputs = (loss.mean(), logits, labels)
        else:
            outputs = (logits,)
        return outputs




class MySimpleModelCheckpoint(SimpleModelCheckpoint):
    def __init__(self,*args,**kwargs):
        super(MySimpleModelCheckpoint, self).__init__(*args,**kwargs)
        self.weight_file = './best.pt'

    def on_save_model(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        pl_module: MyTransformer

        #当前设备
        device = torch.device('cuda:{}'.format(trainer.global_rank))

        if not hasattr(self,'pos_data'):
            self.pos_data = None
            self.neg_data = None

        if self.pos_data is None:
            eval_datasets = dataHelper.load_dataset(dataHelper.eval_files)
            all_data = [eval_datasets[i] for i in range(len(eval_datasets))]
            map_data = {}
            for d in all_data:
                label = np.squeeze(d['labels']).tolist()
                if label not in map_data:
                    map_data[label] = []
                map_data[label].append(d)
            self.pos_data,self.neg_data = generate_pair_example(map_data)

        pos_data = self.pos_data
        neg_data = self.neg_data
        a_data = [_[0] for _ in pos_data + neg_data]
        b_data = [_[1] for _ in pos_data + neg_data]
        labels = np.concatenate([np.ones(len(pos_data),dtype=np.int32),np.zeros(len(neg_data),dtype=np.int32)])

        t_data = a_data + b_data
        from fastdatasets.torch_dataset import Dataset as torch_Dataset
        eval_datasets = DataLoader(torch_Dataset(t_data), batch_size=training_args.eval_batch_size,collate_fn=dataHelper.collate_fn)


        vecs = []
        for i,batch in tqdm(enumerate(eval_datasets),total=len(t_data),desc='evalute'):
            for k in batch:
                batch[k] = batch[k].to(device)
            o = pl_module.validation_step(batch,i)
            b_logits, _ = o['outputs']
            for j in range(len(b_logits)):
                logit = np.asarray(b_logits[j], dtype=np.float32)
                vecs.append(logit)

        a_vecs = np.stack(vecs[:len(a_data)],axis=0)
        b_vecs = np.stack(vecs[len(a_data):],axis=0)

        corrcoef = evaluate_sample(a_vecs,b_vecs,labels)

        f1 = corrcoef

        best_f1 = self.best.get('f1',-np.inf)
        print('current', f1, 'best', best_f1)
        if f1 >= best_f1:
            self.best['f1'] = f1
            logging.info('save best {}, {}\n'.format(self.best['f1'], self.weight_file))
            trainer.save_checkpoint(self.weight_file)


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
    model_args, training_args, data_args = parser.parse_dict(train_info_args)

    checkpoint_callback = MySimpleModelCheckpoint(every_n_train_steps=10000 // training_args.gradient_accumulation_steps)
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
    tokenizer, config, label2id, id2label = load_tokenizer_and_config_with_args(dataHelper, model_args, training_args,
                                                                                data_args)

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

    model = MyTransformer(config=config, model_args=model_args, training_args=training_args)

    if train_datasets is not None:
        trainer.fit(model,train_dataloaders=train_datasets)

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
            trainer.test(model, dataloaders=test_datasets, ckpt_path='./best.pt')


        is_convert_onnx = True
        #是否转换模型
        if is_convert_onnx:
            input_sample = (
                torch.ones(size=(1, 128), dtype=torch.int32),
                torch.ones(size=(1, 128), dtype=torch.int32),
            )
            model.eval()
            model.to('cuda')
            input_names = ["input_ids", "attention_mask"]
            out_names = ["pred_ids"]

            model = MyTransformer.load_from_checkpoint('./best.pt',config=config, model_args=model_args, training_args=training_args)
            model.to_onnx('./best.onnx',
                          input_sample=input_sample,
                          verbose=True,
                          opset_version=10,
                          do_constant_folding=True,
                          input_names=input_names,
                          output_names=out_names,
                          dynamic_axes={"input_ids": [0, 1],
                                        "attention_mask": [0, 1],
                                        "pred_ids": [0, 1]
                                        }
                          )





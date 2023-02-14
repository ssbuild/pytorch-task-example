# -*- coding: utf-8 -*-
import json
import logging
import typing

import numpy as np
import torch
from deep_training.data_helper import DataHelper
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from deep_training.nlp.metrics.pointer import metric_for_spo
from deep_training.nlp.models.gplinker import TransformerForGplinker, extract_spoes
from deep_training.utils.trainer import SimpleModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
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
    # 'train_file': [ '/data/nlp/nlp_train_data/relation/law/step1_train-fastlabel.json'],
    # 'eval_file': [ '/data/nlp/nlp_train_data/relation/law/step1_train-fastlabel.json'],
    # 'label_file': [ '/data/nlp/nlp_train_data/relation/law/relation_label.json'],
    # 'train_file': [ '/data/nlp/nlp_train_data/myrelation/duie/duie_train.json'],
    # 'eval_file': [ '/data/nlp/nlp_train_data/myrelation/duie/duie_dev.json'],
    # 'label_file': [ '/data/nlp/nlp_train_data/myrelation/duie/duie_schema.json'],
    'train_file': [ '/data/nlp/nlp_train_data/myrelation/re_labels.json'],
    'eval_file': [ '/data/nlp/nlp_train_data/myrelation/re_labels.json'],
    'label_file': [ '/data/nlp/nlp_train_data/myrelation/labels.json'],
    'learning_rate': 5e-5,
    'max_epochs': 15,
    'train_batch_size': 20,
    'eval_batch_size': 4,
    'test_batch_size': 2,
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

        sentence, entities, re_list = data
        spo_list = re_list
        tokens = list(sentence) if not do_lower_case else list(sentence.lower())
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[0:(max_seq_length - 2)]
        input_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]'])
        seqlen = len(input_ids)
        attention_mask = [1] * seqlen
        input_ids = np.asarray(input_ids, dtype=np.int32)
        attention_mask = np.asarray(attention_mask, dtype=np.int32)

        max_target_len = 60
        entity_labels = np.zeros(shape=(2, max_target_len, 2), dtype=np.int32)
        head_labels = np.zeros(shape=(len(label2id), max_target_len, 2), dtype=np.int32)
        tail_labels = np.zeros(shape=(len(label2id), max_target_len, 2), dtype=np.int32)

        entity_labels_tmp = [set() for _ in range(2)]
        head_labels_tmp = [set() for _ in range(len(label2id))]
        tail_labels_tmp = [set() for _ in range(len(label2id))]

        real_label = []
        for s, p, o in spo_list:
            p: int = label2id[p]
            real_label.append((s[0], s[1], p, o[0], o[1]))
            s = (s[0] + 1, s[1] + 1)
            o = (o[0] + 1, o[1] + 1)
            if s[1] < max_seq_length - 1 and o[1] < max_seq_length - 1:
                entity_labels_tmp[0].add((s[0], s[1]))
                entity_labels_tmp[1].add((o[0], o[1]))
                head_labels_tmp[p].add((s[0], o[0]))
                tail_labels_tmp[p].add((s[1], o[1]))

        def feed_label(x, pts_list):
            tlens = [1]
            for p, pts in enumerate(pts_list):
                tlens.append(len(pts))
                for seq, pos in enumerate(pts):
                    x[p][seq][0] = pos[0]
                    x[p][seq][1] = pos[1]
            return np.max(tlens)

        targetlen1 = feed_label(entity_labels, list(map(lambda x: list(x), entity_labels_tmp)))
        targetlen2 = feed_label(head_labels, list(map(lambda x: list(x), head_labels_tmp)))
        targetlen3 = feed_label(tail_labels, list(map(lambda x: list(x), tail_labels_tmp)))

        targetlen = np.asarray(np.max([targetlen1, targetlen2, targetlen3]), dtype=np.int32)
        pad_len = max_seq_length - len(input_ids)
        if pad_len > 0:
            pad_val = tokenizer.pad_token_id
            input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            attention_mask = np.pad(attention_mask, (0, pad_len), 'constant', constant_values=(0, 0))
        d = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'entity_labels': entity_labels,
            'head_labels': head_labels,
            'tail_labels': tail_labels,
            'seqlen': seqlen,
            'targetlen': targetlen,
        }

        if self.index < 5:
            print(tokens)
            print(input_ids[:seqlen])

        if mode == 'eval':
            self.eval_labels.append(real_label)
        return d

    # 读取标签
    def on_get_labels(self, files: typing.List):
        labels = []
        label_filename = files[0]
        with open(label_filename, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                jd = json.loads(line)
                if not jd:
                    continue
                larr = [jd['subject'], jd['predicate'], jd['object']]
                labels.append('+'.join(larr))
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

                    entities = jd.get('entities', None)
                    re_list = jd.get('re_list', None)

                    if entities:
                        entities_label = []
                        for k, v in entities.items():
                            pts = [_ for a_ in list(v.values()) for _ in a_]
                            for pt in pts:
                                entities_label.append((k, pt[0], pt[1]))
                    else:
                        entities_label = None

                    if re_list is not None:
                        re_list_label = []
                        for re_node in re_list:
                            for l, relation in re_node.items():
                                s = relation[0]
                                o = relation[1]
                                re_list_label.append((
                                    # (s['pos'][0], s['pos'][1],s['label']),
                                    # l,
                                    # (o['pos'][0], o['pos'][1],o['label'])
                                    (s['pos'][0], s['pos'][1]),
                                    '+'.join([s['label'], l, o['label']]),
                                    (o['pos'][0], o['pos'][1])
                                ))
                    else:
                        re_list_label = None

                    D.append((jd['text'], entities_label, re_list_label))
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
        max_tarlen = torch.max(o.pop('targetlen'))

        o['input_ids'] = o['input_ids'][:, :max_len]
        o['attention_mask'] = o['attention_mask'][:, :max_len]
        if 'token_type_ids' in o:
            o['token_type_ids'] = o['token_type_ids'][:, :max_len]
        o['entity_labels'] = o['entity_labels'][:, :, :max_tarlen]
        o['head_labels'] = o['head_labels'][:, :, :max_tarlen]
        o['tail_labels'] = o['tail_labels'][:, :, :max_tarlen]
        return o


class MyTransformer(TransformerForGplinker, with_pl=True):
    def __init__(self, eval_labels, *args, **kwargs):
        super(MyTransformer, self).__init__(*args, **kwargs)
        self.index = 0
        self.eval_labels = eval_labels

    def validation_epoch_end(self, outputs: typing.Union[EPOCH_OUTPUT, typing.List[EPOCH_OUTPUT]]) -> None:
        self.index += 1
        if self.index < 2:
            self.log('val_f1', 0.0, prog_bar=True)
            return

        threshold = 1e-7
        y_preds, y_trues = [], []
        for i, o in tqdm(enumerate(outputs), total=len(outputs)):
            logits1, logits2, logits3, _, _, _ = o['outputs']
            output_labels = self.eval_labels[i * len(logits1):(i + 1) * len(logits1)]
            p_spoes = extract_spoes([logits1, logits2, logits3], threshold=threshold)
            t_spoes = output_labels
            y_preds.extend(p_spoes)
            y_trues.extend(t_spoes)

        print(y_preds[:3])
        print(y_trues[:3])
        f1, str_report = metric_for_spo(y_trues, y_preds, self.config.label2id)
        print(f1)
        print(str_report)
        self.log('val_f1', f1, prog_bar=True)


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

        eval_labels = pl_module.eval_labels
        config = pl_module.config

        threshold = 1e-7
        y_preds, y_trues = [], []
        for i, batch in tqdm(enumerate(eval_datasets), total=len(eval_datasets), desc='evalute'):
            for k in batch:
                batch[k] = batch[k].to(device)
            o = pl_module.validation_step(batch, i)

            logits1, logits2, logits3, _, _, _ = o['outputs']
            output_labels = eval_labels[i * len(logits1):(i + 1) * len(logits1)]
            p_spoes = extract_spoes([logits1, logits2, logits3], threshold=threshold)
            t_spoes = output_labels
            y_preds.extend(p_spoes)
            y_trues.extend(t_spoes)

        print(y_preds[:3])
        print(y_trues[:3])
        f1, str_report = metric_for_spo(y_trues, y_preds, config.label2id)
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

    checkpoint_callback = MySimpleModelCheckpoint(every_n_epochs=1)
    trainer = Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=training_args.max_epochs,
        max_steps=training_args.max_steps,
        accelerator="gpu",replace_sampler_ddp=False,
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
        dataHelper.make_dataset_with_args(data_args.eval_file, mode='eval')
    if data_args.do_test:
        dataHelper.make_dataset_with_args(data_args.test_file,mode='test')

    model = MyTransformer(dataHelper.eval_labels, with_efficient=False, config=config, model_args=model_args,
                          training_args=training_args)

    if not data_args.convert_onnx:
        train_datasets = dataHelper.load_dataset(dataHelper.train_files, shuffle=True,infinite=True,num_processes=trainer.world_size,process_index=trainer.global_rank)

        if train_datasets is not None:
            train_datasets = DataLoader(train_datasets, batch_size=training_args.train_batch_size,
                                        collate_fn=dataHelper.collate_fn,
                                        shuffle=False if isinstance(train_datasets, IterableDataset) else True)
        if train_datasets is not None:
            trainer.fit(model, train_dataloaders=train_datasets)
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
    else:
        model = MyTransformer.load_from_checkpoint('./best.pt', config=config, model_args=model_args,
                                                   training_args=training_args)
        model.convert_to_onnx('./best.onnx')

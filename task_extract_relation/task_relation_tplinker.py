# -*- coding: utf-8 -*-
import copy
import json
import logging
import typing

import numpy as np
import torch
from deep_training.data_helper import DataHelper
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from deep_training.nlp.metrics.pointer import metric_for_spo
from deep_training.nlp.models.tplinker import TransformerForTplinker, extract_spoes, TplinkerArguments
from deep_training.utils.trainer import SimpleModelCheckpoint
from lightning import Trainer

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
    'learning_rate_for_task': 2e-5,
    'max_epochs': 15,
    'train_batch_size': 6,
    'eval_batch_size': 4,
    'test_batch_size': 2,
    'optimizer': 'adam',
    'adam_epsilon': 1e-8,
    'gradient_accumulation_steps': 1,
    'max_grad_norm': 1.0,
    'weight_decay': 0,
    'warmup_steps': 0,
    'output_dir': './output',
    'train_max_seq_length': 160,
    'eval_max_seq_length': 200,
    'test_max_seq_length': 200,
    # tplinker args
    'shaking_type': 'cln_plus',  # one of ['cat','cat_plus','cln','cln_plus']
    'inner_enc_type': 'linear',  # one of ['mix_pooling','mean_pooling','max_pooling','lstm','linear']
    'dist_emb_size': -1,
    'ent_add_dist': False,
    'rel_add_dist': False,
    # scheduler
    'scheduler_type': 'CAWR',
    'scheduler': {'T_mult': 1, 'rewarm_epoch_num': 2, 'verbose': False},
}


class NN_DataHelper(DataHelper):
    # 是否固定输入最大长度 ， 如果固定训练会慢 ，指标高许多 ，如不固定训练快，指标收敛慢些
    is_fixed_input_length = True

    index = -1
    eval_labels = []

    id2label, label2id = None, None

    # 获取训练集的最大长度
    max_text_length = 0

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
        if mode == 'train':
            max_seq_length = min(max_seq_length, self.max_text_length + 2)

        tokens = list(sentence) if not do_lower_case else list(sentence.lower())
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[0:(max_seq_length - 2)]
        input_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]'])
        seqlen = len(input_ids)
        attention_mask = [1] * seqlen
        input_ids = np.asarray(input_ids, dtype=np.int32)
        attention_mask = np.asarray(attention_mask, dtype=np.int32)

        labels = []
        real_label = []
        for s, p, o in spo_list:
            assert s[0] <= s[1] and o[0] <= o[1]
            p: int = label2id[p]
            real_label.append((s[0], s[1], p, o[0], o[1]))
            s = (s[0] + 1, s[1] + 1)
            o = (o[0] + 1, o[1] + 1)
            if s[1] < max_seq_length - 1 and o[1] < max_seq_length - 1:
                labels.append((s[0], s[1], p, o[0], o[1]))

        labels = np.asarray(labels, dtype=np.int32)
        pad_len = max_seq_length - len(input_ids)
        if pad_len > 0:
            pad_val = tokenizer.pad_token_id
            input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            attention_mask = np.pad(attention_mask, (0, pad_len), 'constant', constant_values=(0, 0))

        # is_fixed_input_length = mode == 'train' and self.is_fixed_input_length
        is_fixed_input_length = self.is_fixed_input_length
        d = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'seqlen': np.asarray(max_seq_length if is_fixed_input_length else seqlen, dtype=np.int32),
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
                                assert s['pos'][0] <= s['pos'][1], ValueError(text, s['pos'])
                                assert o['pos'][0] <= o['pos'][1], ValueError(text, o['pos'])
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
                    text = jd['text']
                    self.max_text_length = max(self.max_text_length, len(text))
                    D.append((text, entities_label, re_list_label))
        return D if mode == 'train' else D[:300]


    def collate_fn(self,batch):
        bs = len(batch)
        o = {}
        spo_labels = []
        for i, b in enumerate(batch):
            b = copy.copy(b)
            spo_labels.append(b.pop('labels', []))
            if i == 0:
                for k in b:
                    o[k] = [torch.tensor(b[k])]
            else:
                for k in b:
                    o[k].append(torch.tensor(b[k]))
        for k in o:
            o[k] = torch.stack(o[k])
        max_len = torch.max(o.pop('seqlen'))
        shaking_len = int(max_len * (max_len + 1) / 2)
        entity_labels = torch.zeros(size=(bs, shaking_len), dtype=torch.long)
        head_labels = torch.zeros(size=(bs, len(self.label2id), shaking_len), dtype=torch.long)
        tail_labels = torch.zeros(size=(bs, len(self.label2id), shaking_len), dtype=torch.long)
        get_pos = lambda x0, x1: x0 * max_len + x1 - int(x0 * (x0 + 1) / 2)
        for spos, e, h, t in zip(spo_labels, entity_labels, head_labels, tail_labels):
            for spo in spos:
                if spo[0] >= max_len - 1 or spo[1] >= max_len - 1 or spo[3] >= max_len - 1 or spo[4] >= max_len - 1:
                    continue
                e[get_pos(spo[0], spo[1])] = 1
                e[get_pos(spo[3], spo[4])] = 1
                if spo[0] <= spo[3]:
                    h[spo[2]][get_pos(spo[0], spo[3])] = 1
                else:
                    h[spo[2]][get_pos(spo[3], spo[0])] = 2
                if spo[1] <= spo[4]:
                    t[spo[2]][get_pos(spo[1], spo[4])] = 1
                else:
                    t[spo[2]][get_pos(spo[4], spo[1])] = 2

        o['input_ids'] = o['input_ids'][:, :max_len]
        o['attention_mask'] = o['attention_mask'][:, :max_len]
        if 'token_type_ids' in o:
            o['token_type_ids'] = o['token_type_ids'][:, :max_len]
        o['entity_labels'] = entity_labels
        o['head_labels'] = head_labels
        o['tail_labels'] = tail_labels
        return o


class MyTransformer(TransformerForTplinker, with_pl=True):
    def __init__(self, eval_labels, *args, **kwargs):
        super(MyTransformer, self).__init__(*args, **kwargs)
        self.index = 0
        self.eval_labels = eval_labels

    # def validation_epoch_end(self, outputs: typing.Union[EPOCH_OUTPUT, typing.List[EPOCH_OUTPUT]]) -> None:
    #     self.index += 1
    #     if self.index < 2:
    #         self.log('val_f1', 0.0, prog_bar=True)
    #         return
    #     y_preds, y_trues = [], []
    #
    #     for i, o in tqdm(enumerate(outputs), total=len(outputs)):
    #         logits1, logits2, logits3, _, _, _ = o['outputs']
    #         bs = len(logits1)
    #         output_labels = self.eval_labels[i * bs:(i + 1) * bs]
    #         p_spoes = extract_spoes([logits1, logits2, logits3])
    #         t_spoes = output_labels
    #         y_preds.extend(p_spoes)
    #         y_trues.extend(t_spoes)
    #
    #     print(y_preds[:3])
    #     print(y_trues[:3])
    #     f1, str_report = metric_for_spo(y_trues, y_preds, self.config.label2id)
    #     print(f1)
    #     print(str_report)
    #     self.log('val_f1', f1, prog_bar=True)


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

        eval_labels = pl_module.eval_labels
        config = pl_module.config

        y_preds, y_trues = [], []
        for i, batch in tqdm(enumerate(eval_datasets), total=len(eval_datasets), desc='evalute'):
            for k in batch:
                batch[k] = batch[k].to(device)
            o = pl_module.validation_step(batch, i)

            logits1, logits2, logits3, _, _, _ = o['outputs']
            bs = len(logits1)
            output_labels = eval_labels[i * bs:(i + 1) * bs]
            p_spoes = extract_spoes([logits1, logits2, logits3])
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
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, TplinkerArguments))
    model_args, training_args, data_args, tplinker_args = parser.parse_dict(train_info_args)

    checkpoint_callback = MySimpleModelCheckpoint(monitor="val_f1", every_n_epochs=1)
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
        dataHelper.make_dataset_with_args(data_args.test_file, shuffle=False,mode='test')


    model = MyTransformer(dataHelper.eval_labels, tplinker_args=tplinker_args, config=config, model_args=model_args,
                          training_args=training_args)

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

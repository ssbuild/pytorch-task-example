# @Time    : 2022/12/17 19:43
# @Author  : tk
# @FileName: numpy_record_reader.py

import numpy as np
import scipy
from fastdatasets.record import load_dataset as Loader, gfile, RECORD, WriterObject


def transform_and_normalize(vecs, kernel=None, bias=None):
    """应用变换，然后标准化
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
    return vecs / np.clip(norms, 1e-8, np.inf)

def compute_corrcoef(x, y):
    """Spearman相关系数
    """
    return scipy.stats.spearmanr(x, y).correlation


if __name__ == '__main__':
    datasets = Loader.IterableDataset('/home/tk/train/eval/eval_vecs.record',options=RECORD.TFRecordOptions(RECORD.TFRecordCompressionType.GZIP))
    datasets = datasets.parse_from_numpy_writer()

    vec_maps = {}
    for i,d in enumerate(datasets):
        l = d['label'].squeeze().tolist()
        if l not in vec_maps:
            vec_maps[l] = []
        vec_maps[l].append(d['logit'])

    labels = []
    a_vecs = []
    b_vecs = []
    for k in vec_maps:
        print(k,len(vec_maps[k]))
        obj_list = vec_maps[k]
        val = [obj_list[ids]
             for ids in
             np.random.choice(list(range(len(obj_list))), min(1000,len(obj_list)))
             ]
        if len(val) > 2:
            for j in range(0,len(val) // 2,2):
                a_vecs.append(val[j])
                b_vecs.append(val[j + 1])
                labels.append(1)

    for k1 in vec_maps.keys():
        for k2 in vec_maps.keys():
            if k1 == k2:
                continue

            obj_list1 = vec_maps[k1]
            val1 = [obj_list1[ids]
                 for ids in
                 np.random.choice(list(range(len(obj_list1))), min(10,len(obj_list1)))
                 ]

            obj_list2 = vec_maps[k2]
            val2 = [obj_list2[ids]
                    for ids in
                    np.random.choice(list(range(len(obj_list2))), min(10, len(obj_list2)))
                    ]

            if val1 and val2:
                for j in range(min(len(val1),len(val2))):
                    a_vecs.append(val1[j])
                    b_vecs.append(val2[j])
                    labels.append(0)


    print('choise')
    print('总样本',len(labels),'正样本',np.sum(labels))

    a_vecs = np.stack(a_vecs, axis=0)
    b_vecs = np.stack(b_vecs, axis=0)
    labels = np.stack(labels, axis=0)


    a_vecs = transform_and_normalize(a_vecs)
    b_vecs = transform_and_normalize(b_vecs)
    sims = (a_vecs * b_vecs).sum(axis=1)
    corrcoef = compute_corrcoef(labels, sims)
    print(corrcoef)

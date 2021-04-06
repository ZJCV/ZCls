# -*- coding: utf-8 -*-

"""
@date: 2021/2/23 下午8:22
@file: imagenet.py
@author: zj
@description: 
"""

import os
import six
import lmdb
import json
import pickle
from PIL import Image
from torch.utils.data import Dataset

from .evaluator.general_evaluator import GeneralEvaluator


def loads_data(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pickle.loads(buf)


def get_imagenet_classes():
    """
    refert to [Imagenet classes](https://discuss.pytorch.org/t/imagenet-classes/4923)
    :return:
    """
    idx2label = []
    # cls2label = {}
    with open("zcls/assets/imagenet_class_index.json", "r") as read_file:
        class_idx = json.load(read_file)
        idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
        # cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}
    return idx2label


class LMDBImageNet(Dataset):

    def __init__(self, root, transform=None, target_transform=None, top_k=(1, 5)):
        assert os.path.isfile(root)

        self.dbpath = root
        self.transform = transform
        self.target_transform = target_transform
        # create evaluator
        self._update_evaluator(top_k)

    def open_lmdb(self):
        self.env = lmdb.open(self.dbpath, subdir=False,
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        # self.env = lmdb.open(self.dbpath, readonly=True, create=False)
        self.txn = self.env.begin(buffers=True)
        self.length = loads_data(self.txn.get(b'__len__'))
        self.keys = loads_data(self.txn.get(b'__keys__'))
        # self.classes = loads_data(self.txn.get(b'classes'))
        self.classes = get_imagenet_classes()

    def __getitem__(self, index: int):
        if not hasattr(self, 'txn'):
            self.open_lmdb()
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        unpacked = loads_data(byteflow)

        # load img
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # return img, target
        return img, target

    def __len__(self) -> int:
        return self.length

    def _update_evaluator(self, top_k):
        self.evaluator = GeneralEvaluator(self.classes, top_k=top_k)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.dbpath + ')'

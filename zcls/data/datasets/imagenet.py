# -*- coding: utf-8 -*-

"""
@date: 2021/2/23 下午8:22
@file: imagenet.py
@author: zj
@description: 
"""

import os
import numpy as np
from PIL import Image
import six
import os.path as osp
import lmdb
import pyarrow as pa
from torch.utils.data import Dataset
import torchvision.datasets as datasets

from .evaluator.general_evaluator import GeneralEvaluator


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


class ImageNet(Dataset):
    """
    [What is the meta.bin file used by the ImageNet dataset? #1646](https://github.com/pytorch/vision/issues/1646)
    torchvision will parse the devkit archive of the ImageNet2012 classification dataset and save
    the meta information in a binary file.
    about problem: TypeError: can't pickle Environment objects
    refert to [TypeError: can't pickle Environment objects when num_workers > 0 for LSUN #689](https://github.com/pytorch/vision/issues/689)
    """

    def __init__(self, root, train=True, transform=None, target_transform=None):
        split = 'train' if train else 'val'
        # using torchvision ImageNet to get classes
        data_set = datasets.ImageNet(root, split=split, transform=transform, target_transform=target_transform)
        self.classes = list()
        for class_tuple in data_set.classes:
            self.classes.append(','.join(class_tuple))
        self.length = len(data_set)

        # get dataset
        self.dbpath = os.path.join(root, f'{split}.lmdb')
        # self.env = lmdb.open(self.dbpath, subdir=osp.isdir(self.dbpath),
        #                      readonly=True, lock=False,
        #                      readahead=False, meminit=False)
        # with self.env.begin(write=False) as txn:
        #     self.length = loads_pyarrow(txn.get(b'__len__'))
        #     self.keys = loads_pyarrow(txn.get(b'__keys__'))
        # get transform and target_transform
        self.transform = transform
        self.target_transform = target_transform
        # create evaluator
        self._update_evaluator()

    def open_lmdb(self):
        self.env = lmdb.open(self.dbpath, subdir=osp.isdir(self.dbpath),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        # self.env = lmdb.open(self.dbpath, readonly=True, create=False)
        self.txn = self.env.begin(buffers=True)
        self.length = loads_pyarrow(self.txn.get(b'__len__'))
        self.keys = loads_pyarrow(self.txn.get(b'__keys__'))

    def __getitem__(self, index: int):
        if not hasattr(self, 'txn'):
            self.open_lmdb()
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        unpacked = loads_pyarrow(byteflow)

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

    def _update_evaluator(self):
        self.evaluator = GeneralEvaluator(self.classes, topk=(1, 5))

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.dbpath + ')'

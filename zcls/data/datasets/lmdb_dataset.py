# -*- coding: utf-8 -*-

"""
@date: 2021/4/4 下午2:55
@file: general_dataset.py
@author: zj
@description: 
"""

import os
import six
import lmdb
import pickle
from PIL import Image
from torch.utils.data import Dataset

from .evaluator.general_evaluator import GeneralEvaluator


def load_data(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pickle.loads(buf)


class LMDBDataset(Dataset):
    """
    [What is the meta.bin file used by the ImageNet dataset? #1646](https://github.com/pytorch/vision/issues/1646)
    torchvision will parse the devkit archive of the ImageNet2012 classification dataset and save
    the meta information in a binary file.
    about problem: TypeError: can't pickle Environment objects
    refert to [TypeError: can't pickle Environment objects when num_workers > 0 for LSUN #689](https://github.com/pytorch/vision/issues/689)
    """

    def __init__(self, root, transform=None, target_transform=None, top_k=(1, 5)):
        assert os.path.isfile(root)

        self.dbpath = root
        self.transform = transform
        self.target_transform = target_transform
        # create evaluator
        self._update_evaluator(top_k)

    def __getitem__(self, index: int):
        if not hasattr(self, 'txn'):
            self.open_lmdb()
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        unpacked = load_data(byteflow)

        # load img
        imgbuf = unpacked[0][0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        target = unpacked[1][0]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        if not hasattr(self, 'txn'):
            self.open_lmdb()
        return self.length

    def _update_evaluator(self, top_k):
        if not hasattr(self, 'txn'):
            self.open_lmdb()
        self.evaluator = GeneralEvaluator(self.classes, top_k=top_k)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.dbpath + ')'

    def open_lmdb(self):
        self.env = lmdb.open(self.dbpath, subdir=False,
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        # self.env = lmdb.open(self.dbpath, readonly=True, create=False)
        self.txn = self.env.begin(buffers=True)
        self.length = load_data(self.txn.get(b'__len__'))
        self.keys = load_data(self.txn.get(b'__keys__'))
        self.classes = self.get_classes(self.txn)

    def get_classes(self, txn):
        return load_data(txn.get(b'classes'))

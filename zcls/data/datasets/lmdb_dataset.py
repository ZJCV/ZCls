# -*- coding: utf-8 -*-

"""
@date: 2021/4/4 下午2:55
@file: general_dataset.py
@author: zj
@description: 
"""

import six
import os
import lmdb
import pickle
from PIL import Image
from torch.utils.data import Dataset

# [Python Pillow - ValueError: Decompressed Data Too Large](https://stackoverflow.com/questions/42671252/python-pillow-valueerror-decompressed-data-too-large)
from PIL import PngImagePlugin

LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024 ** 2)

# [Logs polluted by PIL.PngImagePlugin DEBUG log-level #15](https://github.com/camptocamp/pytest-odoo/issues/15)
import logging

pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)

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
        # with env.begin(write=False) as txn:
        byteflow = self.txn.get(self.keys[index])

        imgbuf, target = load_data(byteflow)
        image = self.get_image(imgbuf)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

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
        self.txn = self.env.begin(write=False, buffers=True)
        self.length = load_data(self.txn.get(b'__len__'))
        self.keys = load_data(self.txn.get(b'__keys__'))
        self.classes = load_data(self.txn.get(b'classes'))

    def get_classes(self):
        if not hasattr(self, 'txn'):
            self.open_lmdb()
        return self.classes

    def get_image(self, imgbuf):
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)

        image = Image.open(buf).convert('RGB')
        return image

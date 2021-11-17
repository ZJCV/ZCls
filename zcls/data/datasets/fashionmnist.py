# -*- coding: utf-8 -*-

"""
@date: 2021/2/23 下午8:22
@file: fashionmnist.py
@author: zj
@description: 
"""

from torch.utils.data import Dataset
import torchvision.datasets as datasets

from .util import default_converter
from .evaluator.general_evaluator import GeneralEvaluator


class FashionMNIST(Dataset):

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 top_k=(1, 5), keep_rgb=False):
        self.data_set = datasets.FashionMNIST(root, train=train, download=True)
        self.classes = self.data_set.classes
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.keep_rgb = keep_rgb
        self._update_evaluator(top_k)

    def __getitem__(self, index: int):
        image, target = self.data_set.__getitem__(index)
        image = default_converter(image, rgb=False)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self) -> int:
        return self.data_set.__len__()

    def _update_evaluator(self, top_k):
        self.evaluator = GeneralEvaluator(self.classes, top_k=top_k)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.root + ')'

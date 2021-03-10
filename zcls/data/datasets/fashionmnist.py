# -*- coding: utf-8 -*-

"""
@date: 2021/2/23 下午8:22
@file: fashionmnist.py
@author: zj
@description: 
"""

from torch.utils.data import Dataset
import torchvision.datasets as datasets

from .evaluator.general_evaluator import GeneralEvaluator


class FashionMNIST(Dataset):

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        self.data_set = datasets.FashionMNIST(root, train=train, transform=transform, target_transform=target_transform,
                                              download=download)
        self.classes = self.data_set.classes
        self._update_evaluator()

    def __getitem__(self, index: int):
        return self.data_set.__getitem__(index)

    def __len__(self) -> int:
        return self.data_set.__len__()

    def _update_evaluator(self):
        self.evaluator = GeneralEvaluator(self.classes, topk=(1, 5))

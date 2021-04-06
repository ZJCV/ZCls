# -*- coding: utf-8 -*-

"""
@date: 2021/2/23 下午8:22
@file: imagenet.py
@author: zj
@description: 
"""

from torch.utils.data import Dataset
import torchvision.datasets as datasets

from .evaluator.general_evaluator import GeneralEvaluator


class ImageNet(Dataset):

    def __init__(self, root, train=True, transform=None, target_transform=None, top_k=(1, 5)):
        split = 'train' if train else 'val'
        # using torchvision ImageNet to get classes
        self.data_set = datasets.ImageNet(root, split=split, transform=transform, target_transform=target_transform)
        self.classes = list()
        for class_tuple in self.data_set.classes:
            self.classes.append(','.join(class_tuple))
        self.root = root
        # create evaluator
        self._update_evaluator(top_k)

    def __getitem__(self, index: int):
        return self.data_set.__getitem__(index)

    def __len__(self) -> int:
        return len(self.data_set)

    def _update_evaluator(self, top_k):
        self.evaluator = GeneralEvaluator(self.classes, top_k=top_k)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.root + ')'

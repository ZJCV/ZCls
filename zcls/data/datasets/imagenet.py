# -*- coding: utf-8 -*-

"""
@date: 2021/2/23 下午8:22
@file: imagenet.py
@author: zj
@description: 
"""

from torch.utils.data import Dataset
import torchvision.datasets as datasets

from .util import default_converter
from .evaluator.general_evaluator import GeneralEvaluator


class ImageNet(Dataset):

    def __init__(self, root, train=True, transform=None, target_transform=None, top_k=(1, 5), keep_rgb=False):
        split = 'train' if train else 'val'
        # using torchvision ImageNet to get classes
        self.data_set = datasets.ImageNet(root, split=split)
        self.classes = list()
        for class_tuple in self.data_set.classes:
            self.classes.append(','.join(class_tuple))
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.keep_rgb = keep_rgb
        # create evaluator
        self._update_evaluator(top_k)

    def __getitem__(self, index: int):
        image, target = self.data_set.__getitem__(index)
        image = default_converter(image, rgb=self.keep_rgb)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self) -> int:
        return len(self.data_set)

    def _update_evaluator(self, top_k):
        self.evaluator = GeneralEvaluator(self.classes, top_k=top_k)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.root + ')'

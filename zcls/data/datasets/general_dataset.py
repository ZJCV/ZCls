# -*- coding: utf-8 -*-

"""
@date: 2021/4/4 下午2:55
@file: general_dataset.py
@author: zj
@description: 
"""

from torch.utils.data import Dataset
import torchvision.datasets as datasets

from .util import default_converter
from .evaluator.general_evaluator import GeneralEvaluator


class GeneralDataset(Dataset):

    def __init__(self, root, transform=None, target_transform=None, top_k=(1, 5), keep_rgb=False):
        self.data_set = datasets.ImageFolder(root)
        self.classes = self.data_set.classes
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.keep_rgb = keep_rgb
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
        return self.data_set.__len__()

    def _update_evaluator(self, top_k):
        self.evaluator = GeneralEvaluator(self.classes, top_k=top_k)

    def get_classes(self):
        return self.classes

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.root + ')'

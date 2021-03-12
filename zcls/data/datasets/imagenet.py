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
    """
    [What is the meta.bin file used by the ImageNet dataset? #1646](https://github.com/pytorch/vision/issues/1646)
    torchvision will parse the devkit archive of the ImageNet2012 classification dataset and save
    the meta information in a binary file.
    """

    def __init__(self, root, train=True, transform=None, target_transform=None):
        split = 'train' if train else 'val'

        self.data_set = datasets.ImageNet(root, split=split, transform=transform, target_transform=target_transform)
        self.classes = list()
        for class_tuple in self.data_set.classes:
            self.classes.append(','.join(class_tuple))
        self._update_evaluator()

    def __getitem__(self, index: int):
        return self.data_set.__getitem__(index)

    def __len__(self) -> int:
        return self.data_set.__len__()

    def _update_evaluator(self):
        self.evaluator = GeneralEvaluator(self.classes, topk=(1, 5))

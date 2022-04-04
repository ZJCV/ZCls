# -*- coding: utf-8 -*-

"""
@date: 2022/4/4 上午11:04
@file: general_dataset.py
@author: zj
@description: 
"""

from torch.utils.data import Dataset
import torchvision.datasets as datasets


class GeneralDataset(Dataset):

    def __init__(self, root, transform=None, target_transform=None):
        self.data_set = datasets.ImageFolder(root, transform=transform, target_transform=target_transform)
        self.classes = self.data_set.classes
        self.root = root

    def __getitem__(self, index: int):
        image, target = self.data_set.__getitem__(index)

        return image, target

    def __len__(self) -> int:
        return self.data_set.__len__()

    def get_classes(self):
        return self.classes

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.root + ')'

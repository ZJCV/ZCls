# -*- coding: utf-8 -*-

"""
@date: 2020/8/22 下午9:21
@file: trainer.py
@author: zj
@description: 
"""

from .cifar import CIFAR
from .fashionmnist import FashionMNIST
from .imagenet import ImageNet


def build_dataset(cfg, transform=None, is_train=True, download=True):
    dataset_name = cfg.DATASET.NAME
    data_dir = cfg.DATASET.DATA_DIR

    if dataset_name == 'CIFAR100':
        dataset = CIFAR(data_dir, train=is_train, transform=transform, download=download, is_cifar100=True)
    elif dataset_name == 'CIFAR10':
        dataset = CIFAR(data_dir, train=is_train, transform=transform, download=download, is_cifar100=False)
    elif dataset_name == 'FashionMNIST':
        dataset = FashionMNIST(data_dir, train=is_train, transform=transform, download=download)
    elif dataset_name == 'ImageNet':
        dataset = ImageNet(data_dir, train=is_train, transform=transform)
    else:
        raise ValueError(f"the dataset {dataset_name} does not exist")

    return dataset

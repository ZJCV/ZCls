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
from .general_dataset import GeneralDataset
from .lmdb_dataset import LMDBDataset
from .lmdb_imagenet import LMDBImageNet


def build_dataset(cfg, transform=None, target_transform=None, is_train=True):
    dataset_name = cfg.DATASET.NAME
    data_root = cfg.DATASET.TRAIN_ROOT if is_train else cfg.DATASET.TEST_ROOT
    top_k = cfg.DATASET.TOP_K

    if dataset_name == 'CIFAR100':
        dataset = CIFAR(data_root, train=is_train, transform=transform, target_transform=target_transform,
                        top_k=top_k, is_cifar100=True)
    elif dataset_name == 'CIFAR10':
        dataset = CIFAR(data_root, train=is_train, transform=transform, target_transform=target_transform,
                        top_k=top_k, is_cifar100=False)
    elif dataset_name == 'FashionMNIST':
        dataset = FashionMNIST(data_root, train=is_train, transform=transform, target_transform=target_transform,
                               top_k=top_k)
    elif dataset_name == 'ImageNet':
        dataset = ImageNet(data_root, train=is_train, transform=transform, target_transform=target_transform,
                           top_k=top_k)
    elif dataset_name == 'GeneralDataset':
        dataset = GeneralDataset(data_root, transform=transform, target_transform=target_transform, top_k=top_k)
    elif dataset_name == 'LMDBDataset':
        dataset = LMDBDataset(data_root, transform=transform, target_transform=target_transform, top_k=top_k)
    elif dataset_name == 'LMDBImageNet':
        dataset = LMDBImageNet(data_root, transform=transform, target_transform=target_transform, top_k=top_k)
    else:
        raise ValueError(f"the dataset {dataset_name} does not exist")

    return dataset

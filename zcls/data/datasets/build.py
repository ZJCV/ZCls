# -*- coding: utf-8 -*-

"""
@date: 2020/8/22 下午9:21
@file: trainer.py
@author: zj
@description: 
"""

import torch

from .cifar import CIFAR
from .fashionmnist import FashionMNIST
from .imagenet import ImageNet
from .general_dataset import GeneralDataset
from .general_dataset_v2 import GeneralDatasetV2
from .lmdb_dataset import LMDBDataset
from .lmdb_imagenet import LMDBImageNet
from .mp_dataset import MPDataset


def build_dataset(cfg, transform=None, target_transform=None, is_train=True, **kwargs):
    dataset_name = cfg.DATASET.NAME
    data_root = cfg.DATASET.TRAIN_ROOT if is_train else cfg.DATASET.TEST_ROOT
    top_k = cfg.DATASET.TOP_K
    keep_rgb = cfg.DATASET.KEEP_RGB

    if dataset_name == 'CIFAR100':
        dataset = CIFAR(data_root, train=is_train, transform=transform, target_transform=target_transform,
                        top_k=top_k, is_cifar100=True, keep_rgb=keep_rgb)
    elif dataset_name == 'CIFAR10':
        dataset = CIFAR(data_root, train=is_train, transform=transform, target_transform=target_transform,
                        top_k=top_k, is_cifar100=False, keep_rgb=keep_rgb)
    elif dataset_name == 'FashionMNIST':
        dataset = FashionMNIST(data_root, train=is_train, transform=transform, target_transform=target_transform,
                               top_k=top_k, keep_rgb=keep_rgb)
    elif dataset_name == 'ImageNet':
        dataset = ImageNet(data_root, train=is_train, transform=transform, target_transform=target_transform,
                           top_k=top_k, keep_rgb=keep_rgb)
    elif dataset_name == 'GeneralDataset':
        dataset = GeneralDataset(data_root, transform=transform, target_transform=target_transform, top_k=top_k,
                                 keep_rgb=keep_rgb)
    elif dataset_name == 'GeneralDatasetV2':
        dataset = GeneralDatasetV2(data_root, transform=transform, target_transform=target_transform, top_k=top_k,
                                   keep_rgb=keep_rgb)
    elif dataset_name == 'LMDBDataset':
        dataset = LMDBDataset(data_root, transform=transform, target_transform=target_transform, top_k=top_k,
                              keep_rgb=keep_rgb)
    elif dataset_name == 'LMDBImageNet':
        dataset = LMDBImageNet(data_root, transform=transform, target_transform=target_transform, top_k=top_k,
                               keep_rgb=keep_rgb)
    elif dataset_name == 'MPDataset':
        shuffle = cfg.DATALOADER.RANDOM_SAMPLE if is_train else False
        num_gpus = cfg.NUM_GPUS
        rank_id = kwargs.get('rank_id', 0)
        epoch = kwargs.get('epoch', 0)

        dataset = MPDataset(data_root, transform=transform, target_transform=target_transform, top_k=top_k,
                            keep_rgb=keep_rgb, shuffle=shuffle, num_gpus=num_gpus, rank_id=rank_id, epoch=epoch,
                            drop_last=is_train)
    else:
        raise ValueError(f"the dataset {dataset_name} does not exist")

    return dataset

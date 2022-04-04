# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午4:46
@file: build.py
@author: zj
@description: 
"""

import os

from ..transform.build import build_transform
from .general_dataset import GeneralDataset
from .general_dataset_v2 import GeneralDatasetV2
from .mp_dataset import MPDataset

__supported_dataset__ = [
    'general',
    'general_v2',
    'mp'
]


def build_dataset(args):
    assert args.dataset in __supported_dataset__, f"{args.dataset} do not support"
    train_transform, val_transform = build_transform(args)

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    # valdir = os.path.join(args.data, 'val')
    valdir = os.path.join(args.data, 'test')

    if args.dataset == 'general':
        train_dataset = GeneralDataset(
            traindir, transform=train_transform
        )
        val_dataset = GeneralDataset(
            valdir, transform=val_transform
        )
    elif args.dataset == 'general_v2':
        train_dataset = GeneralDatasetV2(
            traindir, transform=train_transform
        )
        val_dataset = GeneralDatasetV2(
            valdir, transform=val_transform
        )
    elif args.dataset == 'mp':
        num_gpus = args.world_size
        rank_id = args.local_rank
        epoch = args.epoch

        train_dataset = MPDataset(
            traindir, transform=train_transform, shuffle=True, num_gpus=num_gpus, rank_id=rank_id, epoch=epoch
        )
        val_dataset = MPDataset(
            valdir, transform=val_transform, shuffle=False, num_gpus=num_gpus, rank_id=rank_id, epoch=epoch
        )
    else:
        raise ValueError(f"{args.dataset} do not support")

    return train_dataset, val_dataset

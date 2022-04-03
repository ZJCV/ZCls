# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午4:47
@file: build.py
@author: zj
@description: 
"""

from torch.utils.data.distributed import DistributedSampler


def build_sampler(args, train_dataset, val_dataset):
    train_sampler = None
    val_sampler = None
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset)

    return train_sampler, val_sampler

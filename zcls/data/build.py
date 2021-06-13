# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午7:20
@file: build.py
@author: zj
@description: 
"""

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler

from .datasets.build import build_dataset
from .transforms.build import build_transform
from .dataloader.build import build_dataloader


def build_data(cfg, is_train=True):
    transform, target_transform = build_transform(cfg, is_train=is_train)
    dataset = build_dataset(cfg, transform=transform, target_transform=target_transform, is_train=is_train)

    return build_dataloader(cfg, dataset, is_train=is_train)


def shuffle_dataset(loader, cur_epoch, is_shuffle=False):
    """"
    Shuffles the data.
    Args:
        loader (loader): data loader to perform shuffle.
        cur_epoch (int): number of the current epoch.
        is_shuffle (bool): need to shuffle the data
    """
    if not is_shuffle:
        return
    sampler = loader.sampler
    assert isinstance(
        sampler, (RandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        sampler.set_epoch(cur_epoch)

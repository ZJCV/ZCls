# -*- coding: utf-8 -*-

"""
@date: 2021/3/31 上午11:25
@file: build.py
@author: zj
@description: 
"""

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler, SequentialSampler

import zcls.util.distributed as du

from ..datasets.mp_dataset import MPDataset


def build_sampler(cfg, dataset):
    world_size = du.get_world_size()
    num_gpus = cfg.NUM_GPUS
    rank = du.get_rank()

    if num_gpus <= 1:
        if cfg.DATALOADER.RANDOM_SAMPLE:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
    else:
        shuffle = cfg.DATALOADER.RANDOM_SAMPLE
        sampler = DistributedSampler(dataset,
                                     num_replicas=world_size,
                                     rank=rank,
                                     shuffle=shuffle)

    return sampler


def build_dataloader(cfg, dataset, is_train=True, **kwargs):
    device_type = kwargs.get('device_type', 'cuda')
    batch_size = cfg.DATALOADER.TRAIN_BATCH_SIZE if is_train else cfg.DATALOADER.TEST_BATCH_SIZE

    sampler = None if isinstance(dataset, MPDataset) else build_sampler(cfg, dataset)
    data_loader = DataLoader(dataset,
                             num_workers=cfg.DATALOADER.NUM_WORKERS,
                             sampler=sampler,
                             batch_size=batch_size,
                             drop_last=is_train,
                             # [When to set pin_memory to true?](https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723)
                             # Just using CPU, set pin_memory=False. Default: True
                             pin_memory=True if 'cuda' in device_type else False)

    return data_loader

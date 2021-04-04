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


def build_dataloader(cfg, dataset, is_train=True):
    world_size = du.get_world_size()
    num_gpus = cfg.NUM_GPUS
    rank = du.get_rank()
    if is_train:
        batch_size = cfg.DATALOADER.TRAIN_BATCH_SIZE

        if num_gpus > 1:
            sampler = DistributedSampler(dataset,
                                         num_replicas=world_size,
                                         rank=rank,
                                         shuffle=True)
        else:
            sampler = RandomSampler(dataset)
    else:
        batch_size = cfg.DATALOADER.TEST_BATCH_SIZE

        if num_gpus > 1:
            sampler = DistributedSampler(dataset,
                                         num_replicas=world_size,
                                         rank=rank,
                                         shuffle=False)
        else:
            sampler = SequentialSampler(dataset)

    data_loader = DataLoader(dataset,
                             num_workers=cfg.DATALOADER.NUM_WORKERS,
                             sampler=sampler,
                             batch_size=batch_size,
                             drop_last=False,
                             # [When to set pin_memory to true?](https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723)
                             pin_memory=True)

    return data_loader

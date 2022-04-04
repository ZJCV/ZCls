# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午1:30
@file: build.py
@author: zj
@description: 
"""

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.data import IterableDataset
import torch.utils.data.distributed

from .dataset.build import build_dataset
from .sampler.build import build_sampler
from .dataloader.collate import fast_collate


def build_data(args, memory_format):
    train_dataset, val_dataset = build_dataset(args)

    if isinstance(train_dataset, IterableDataset):
        train_sampler, val_sampler = None, None
        shuffle = False
    else:
        train_sampler, val_sampler = build_sampler(args, train_dataset, val_dataset)
        shuffle = train_sampler is None

    collate_fn = lambda b: fast_collate(b, memory_format)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        sampler=val_sampler,
        collate_fn=collate_fn)

    return train_sampler, train_loader, val_loader

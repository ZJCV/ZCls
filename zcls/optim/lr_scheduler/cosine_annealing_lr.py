# -*- coding: utf-8 -*-

"""
@date: 2022/4/4 下午10:24
@file: cosine_annealing_lr.py
@author: zj
@description: 
"""

import torch.optim as optim
from torch.optim.optimizer import Optimizer


def build_cosine_annearling_lr(args, optimizer):
    assert isinstance(optimizer, Optimizer)

    max_epoch = args.epochs
    if args.warmup:
        max_epoch = max_epoch - args.warmup_epochs
    minimal_lr = 1e-6
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=minimal_lr)
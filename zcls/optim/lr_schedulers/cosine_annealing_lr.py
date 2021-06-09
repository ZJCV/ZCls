# -*- coding: utf-8 -*-

"""
@date: 2020/9/9 下午9:49
@file: cosine_annealing_lr.py
@author: zj
@description: 
"""

import torch.optim as optim
from torch.optim.optimizer import Optimizer

from .. import registry


@registry.LR_SCHEDULERS.register('CosineAnnealingLR')
def build_cosine_annealing_lr(cfg, optimizer):
    assert isinstance(optimizer, Optimizer)

    max_epoch = cfg.TRAIN.MAX_EPOCH
    if cfg.LR_SCHEDULER.IS_WARMUP:
        max_epoch -= cfg.LR_SCHEDULER.WARMUP.ITERATION
    minimal_lr = cfg.LR_SCHEDULER.COSINE_ANNEALING_LR.MINIMAL_LR

    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=minimal_lr)

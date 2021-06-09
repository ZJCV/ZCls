# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午7:55
@file: trainer.py
@author: zj
@description: 
"""

from torch.optim.optimizer import Optimizer

from .. import registry
from .gradual_warmup import GradualWarmupScheduler
from .multistep_lr import build_multistep_lr
from .cosine_annealing_lr import build_cosine_annealing_lr


def build_lr_scheduler(cfg, optimizer):
    assert isinstance(optimizer, Optimizer)
    lr_scheduler = registry.LR_SCHEDULERS[cfg.LR_SCHEDULER.NAME](cfg, optimizer)

    if cfg.LR_SCHEDULER.IS_WARMUP:
        lr_scheduler = GradualWarmupScheduler(optimizer,
                                              multiplier=cfg.LR_SCHEDULER.WARMUP.MULTIPLIER,
                                              total_epoch=cfg.LR_SCHEDULER.WARMUP.ITERATION,
                                              after_scheduler=lr_scheduler)

        optimizer.zero_grad()
        optimizer.step()
        lr_scheduler.step()

    return lr_scheduler

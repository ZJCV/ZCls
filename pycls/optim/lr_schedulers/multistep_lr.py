# -*- coding: utf-8 -*-

"""
@date: 2020/8/27 下午2:28
@file: multistep_lr.py
@author: zj
@description: 
"""

import torch.optim as optim
from torch.optim.optimizer import Optimizer

from .. import registry


@registry.LR_SCHEDULERS.register('MultiStepLR')
def build_multistep_lr(cfg, optimizer):
    assert isinstance(optimizer, Optimizer)

    milestones = cfg.LR_SCHEDULER.MULTISTEP_LR.MILESTONES
    gamma = cfg.LR_SCHEDULER.GAMMA

    return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

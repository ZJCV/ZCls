# -*- coding: utf-8 -*-

"""
@date: 2022/4/4 下午10:42
@file: multi_step_lr.py
@author: zj
@description: 
"""

import torch.optim as optim
from torch.optim.optimizer import Optimizer


def build_multistep_lr(args, optimizer):
    assert isinstance(optimizer, Optimizer)

    milestones = [30, 60, 80]
    gamma = 0.1
    return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

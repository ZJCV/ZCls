# -*- coding: utf-8 -*-

"""
@date: 2020/8/22 下午8:55
@file: sgd.py
@author: zj
@description: 
"""

import torch.nn as nn
import torch.optim as optim

from .. import registry


@registry.OPTIMIZERS.register('SGD')
def build_sgd(cfg, groups):
    lr = cfg.OPTIMIZER.LR
    weight_decay = cfg.OPTIMIZER.WEIGHT_DECAY.DECAY

    momentum = cfg.OPTIMIZER.MOMENTUM

    return optim.SGD(groups, lr=lr, momentum=momentum, weight_decay=weight_decay)

# -*- coding: utf-8 -*-

"""
@date: 2021/1/1 下午6:51
@file: rmsprop.py
@author: zj
@description: 
"""

import torch.optim as optim

from .. import registry


@registry.OPTIMIZERS.register('RMSProp')
def build_rmsprop(cfg, groups):
    lr = cfg.OPTIMIZER.LR
    weight_decay = cfg.OPTIMIZER.WEIGHT_DECAY.DECAY

    momentum = cfg.OPTIMIZER.MOMENTUM

    return optim.RMSprop(groups, lr=lr, momentum=momentum, weight_decay=weight_decay)

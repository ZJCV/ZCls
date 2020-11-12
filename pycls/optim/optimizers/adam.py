# -*- coding: utf-8 -*-

"""
@date: 2020/8/22 下午9:02
@file: adam.py
@author: zj
@description: 
"""

import torch.nn as nn
import torch.optim as optim

from .. import registry


@registry.OPTIMIZERS.register('ADAM')
def build_adam(cfg, model):
    assert isinstance(model, nn.Module)

    lr = cfg.OPTIMIZER.LR
    weight_decay = cfg.OPTIMIZER.WEIGHT_DECAY

    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

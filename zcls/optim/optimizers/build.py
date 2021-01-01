# -*- coding: utf-8 -*-

"""
@date: 2020/11/4 下午1:49
@file: build.py
@author: zj
@description: 
"""

import torch.nn as nn

from .. import registry
from .sgd import build_sgd
from .adam import build_adam
from .rmsprop import build_rmsprop


def build_optimizer(cfg, model):
    assert isinstance(model, nn.Module)
    groups = group_weight(model)
    optimizer = registry.OPTIMIZERS[cfg.OPTIMIZER.NAME](cfg, groups)
    optimizer.zero_grad()

    return optimizer


def group_weight(module):
    """
    参考[Allow to set 0 weight decay for biases and params in batch norm #1402](https://github.com/pytorch/pytorch/issues/1402)
    过滤所有层bias和归一化层用于权重衰减
    """
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, (nn.modules.batchnorm._BatchNorm, nn.GroupNorm, nn.LayerNorm)):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups

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


def build_optimizer(cfg, model):
    assert isinstance(model, nn.Module)
    optimizer = registry.OPTIMIZERS[cfg.OPTIMIZER.NAME](cfg, model)
    optimizer.zero_grad()

    return optimizer

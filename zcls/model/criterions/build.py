# -*- coding: utf-8 -*-

"""
@date: 2020/11/4 下午1:59
@file: build.py
@author: zj
@description: 
"""

from .. import registry
from .crossentropy_loss import CrossEntropyLoss
from .label_smoothing_loss import LabelSmoothingLoss


def build_criterion(cfg, device):
    return registry.CRITERION[cfg.MODEL.CRITERION.NAME](cfg).to(device=device)

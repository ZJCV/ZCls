# -*- coding: utf-8 -*-

"""
@date: 2020/11/25 下午6:49
@file: optimizer.py
@author: zj
@description: 
"""

from yacs.config import CfgNode as CN


def add_config(_C):
    # ---------------------------------------------------------------------------- #
    # Optimizer
    # ---------------------------------------------------------------------------- #
    _C.OPTIMIZER = CN()
    _C.OPTIMIZER.NAME = 'SGD'
    _C.OPTIMIZER.LR = 1e-3
    _C.OPTIMIZER.MOMENTUM = 0.9
    # ---------------------------------------------------------------------------- #
    # Weight Decay
    # ---------------------------------------------------------------------------- #
    _C.OPTIMIZER.WEIGHT_DECAY = CN()
    _C.OPTIMIZER.WEIGHT_DECAY.DECAY = 1e-4
    _C.OPTIMIZER.WEIGHT_DECAY.NO_BIAS = False
    _C.OPTIMIZER.WEIGHT_DECAY.NO_NORM = False

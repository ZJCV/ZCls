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
    _C.OPTIMIZER.WEIGHT_DECAY = 3e-5
    # for sgd
    _C.OPTIMIZER.SGD = CN()
    _C.OPTIMIZER.SGD.MOMENTUM = 0.9

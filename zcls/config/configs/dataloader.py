# -*- coding: utf-8 -*-

"""
@date: 2020/11/25 下午6:50
@file: dataloader.py
@author: zj
@description: 
"""

from yacs.config import CfgNode as CN


def add_config(_C):
    # ---------------------------------------------------------------------------- #
    # DataLoader
    # ---------------------------------------------------------------------------- #
    _C.DATALOADER = CN()
    _C.DATALOADER.TRAIN_BATCH_SIZE = 16
    _C.DATALOADER.TEST_BATCH_SIZE = 16
    _C.DATALOADER.NUM_WORKERS = 8

# -*- coding: utf-8 -*-

"""
@date: 2020/11/25 下午6:52
@file: dataset.py
@author: zj
@description: 
"""

from yacs.config import CfgNode as CN


def add_config(_C):
    # ---------------------------------------------------------------------------- #
    # DataSet
    # ---------------------------------------------------------------------------- #
    _C.DATASET = CN()
    _C.DATASET.NAME = 'CIFAR100'
    _C.DATASET.DATA_DIR = './data/cifar'

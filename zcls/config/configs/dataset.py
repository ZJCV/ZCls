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
    # train data path
    _C.DATASET.TRAIN_ROOT = './data/cifar'
    # test data path
    _C.DATASET.TEST_ROOT = './data/cifar'
    # used for evaluator
    _C.DATASET.TOP_K = (1,)

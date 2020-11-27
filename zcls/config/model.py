# -*- coding: utf-8 -*-

"""
@date: 2020/11/25 下午6:49
@file: model.py
@author: zj
@description: 
"""

from yacs.config import CfgNode as CN


def add_config(_C):
    # ---------------------------------------------------------------------------- #
    # Model
    # ---------------------------------------------------------------------------- #
    _C.MODEL = CN()
    _C.MODEL.NAME = 'ResNet'
    _C.MODEL.PRETRAINED = ""
    _C.MODEL.TORCHVISION_PRETRAINED = False
    _C.MODEL.SYNC_BN = False
    _C.MODEL.GROUPS = 3

    _C.MODEL.BACKBONE = CN()
    # for ResNet
    _C.MODEL.BACKBONE.ARCH = 'resnet18'

    _C.MODEL.HEAD = CN()
    _C.MODEL.HEAD.FEATURE_DIMS = 2048
    _C.MODEL.HEAD.NUM_CLASSES = 1000

    _C.MODEL.RECOGNIZER = CN()
    _C.MODEL.RECOGNIZER.NAME = 'ResNet_Pytorch'

    _C.MODEL.CRITERION = CN()
    _C.MODEL.CRITERION.NAME = 'CrossEntropyLoss'

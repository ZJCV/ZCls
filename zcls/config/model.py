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

    _C.MODEL.NORM = CN()
    _C.MODEL.NORM.TYPE = 'BatchNorm2d'
    # for bn
    _C.MODEL.NORM.SYNC_BN = False
    _C.MODEL.NORM.FIX_BN = False
    _C.MODEL.NORM.PARTIAL_BN = False
    # Precise BN stats.
    _C.MODEL.NORM.PRECISE_BN = False
    # Number of samples use to compute precise bn.
    _C.MODEL.NORM.NUM_BATCHES_PRECISE = 200
    # for groupnorm
    _C.MODEL.NORM.GROUPS = 32

    _C.MODEL.ACT = CN()
    _C.MODEL.ACT.TYPE = 'ReLU'

    _C.MODEL.COMPRESSION = CN()
    _C.MODEL.COMPRESSION.WIDTH_MULTIPLIER = 1.0

    _C.MODEL.ATTENTION = CN()
    _C.MODEL.ATTENTION.WITH_ATTENTION = (1, 1, 1, 1)
    _C.MODEL.ATTENTION.REDUCTION = 16
    _C.MODEL.ATTENTION.ATTENTION_TYPE = 'GlobalContextBlock2D'

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

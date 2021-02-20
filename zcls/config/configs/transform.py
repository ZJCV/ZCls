# -*- coding: utf-8 -*-

"""
@date: 2020/11/25 下午6:51
@file: transform.py
@author: zj
@description: 
"""

from yacs.config import CfgNode as CN
from ztransforms.cls.autoaugment import AutoAugmentPolicy


def add_config(_C):
    # ---------------------------------------------------------------------------- #
    # Transform
    # ---------------------------------------------------------------------------- #
    _C.TRANSFORM = CN()
    _C.TRANSFORM.MEAN = (0.45, 0.45, 0.45)
    _C.TRANSFORM.STD = (0.225, 0.225, 0.225)

    # ---------------------------------------------------------------------------- #
    # Train
    # ---------------------------------------------------------------------------- #
    _C.TRANSFORM.TRAIN = CN()

    # resize
    _C.TRANSFORM.TRAIN.RESIZE = True
    _C.TRANSFORM.TRAIN.SHORTER_SIDE = 224

    # crop
    _C.TRANSFORM.TRAIN.CENTER_CROP = True
    _C.TRANSFORM.TRAIN.RANDOM_CROP = False
    _C.TRANSFORM.TRAIN.TRAIN_CROP_SIZE = 224

    # random horizontal flip
    _C.TRANSFORM.TRAIN.RANDOM_HORIZONTAL_FLIP = False

    # color jitter
    _C.TRANSFORM.TRAIN.WITH_COLOR_JITTING = False
    # (brightness, contrast, saturation, hue)
    _C.TRANSFORM.TRAIN.COLOR_JITTING = (0.1, 0.1, 0.1, 0.1)

    # auto augment
    _C.TRANSFORM.TRAIN.AUTO_AUGMENT = False
    _C.TRANSFORM.TRAIN.AUGMENT_POLICY = AutoAugmentPolicy.IMAGENET.value

    # ---------------------------------------------------------------------------- #
    # Test
    # ---------------------------------------------------------------------------- #
    _C.TRANSFORM.TEST = CN()

    # resize
    _C.TRANSFORM.TEST.RESIZE = True
    _C.TRANSFORM.TEST.SHORTER_SIDE = 224

    # crop
    _C.TRANSFORM.TEST.CENTER_CROP = True
    _C.TRANSFORM.TEST.TEST_CROP_SIZE = 224

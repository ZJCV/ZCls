# -*- coding: utf-8 -*-

"""
@date: 2020/11/25 下午6:51
@file: transform.py
@author: zj
@description: 
"""


from yacs.config import CfgNode as CN


def add_config(_C):
    # ---------------------------------------------------------------------------- #
    # Transform
    # ---------------------------------------------------------------------------- #
    _C.TRANSFORM = CN()
    _C.TRANSFORM.MEAN = (0.5071, 0.4865, 0.4409)
    _C.TRANSFORM.STD = (0.1942, 0.1918, 0.1958)

    _C.TRANSFORM.TRAIN = CN()
    _C.TRANSFORM.TRAIN.SHORTER_SIDE = 224
    _C.TRANSFORM.TRAIN.CENTER_CROP = True
    _C.TRANSFORM.TRAIN.TRAIN_CROP_SIZE = 224

    _C.TRANSFORM.TEST = CN()
    _C.TRANSFORM.TEST.SHORTER_SIDE = 224
    _C.TRANSFORM.TEST.CENTER_CROP = True
    _C.TRANSFORM.TEST.TEST_CROP_SIZE = 224
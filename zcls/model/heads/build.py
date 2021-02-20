# -*- coding: utf-8 -*-

"""
@date: 2021/2/19 下午1:43
@file: build.py
@author: zj
@description: 
"""

from .. import registry

from .general_head_2d import build_general_head_2d
from .general_head_3d import build_general_head_3d
from .mobilenetv3_head import build_mbv3_head


def build_head(cfg):
    return registry.HEAD[cfg.MODEL.HEAD.NAME](cfg)

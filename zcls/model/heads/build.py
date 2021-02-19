# -*- coding: utf-8 -*-

"""
@date: 2021/2/19 下午1:43
@file: build.py
@author: zj
@description: 
"""

from .. import registry

from .general_head_2d import build_general_head_2d


def build_head(cfg):
    return registry.HEAD[cfg.MODEL.HEAD.NAME](cfg)

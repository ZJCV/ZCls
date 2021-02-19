# -*- coding: utf-8 -*-

"""
@date: 2021/2/19 下午1:43
@file: build.py
@author: zj
@description: 
"""

from .. import registry

from .shufflenet.shufflenetv1_backbone import build_sfv1_backbone
from .shufflenet.shufflenetv2_backbone import build_sfv2_backbone


def build_backbone(cfg):
    return registry.Backbone[cfg.MODEL.BACKBONE.NAME](cfg)

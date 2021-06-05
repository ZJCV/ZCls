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
from .mobilenet.mobilenetv1_backbone import build_mbv1_backbone
from .mobilenet.mobilenetv2_backbone import build_mbv2_backbone
from .mobilenet.mnasnet_backbone import build_mnasnet
from .mobilenet.mobilenetv3_backbone import build_mbv3_backbone
from .vgg.repvgg_backbone import build_repvgg_backbone
from .resnet.resnet_backbone import build_resnet_backbone
from .resnet.resnet_d_backbone import build_resnet_d_backbone
from .resnet.resnet3d_backbone import build_resnet3d_backbone
from .ghostnet.ghost_backbone import build_ghostnet_backbone


def build_backbone(cfg):
    return registry.Backbone[cfg.MODEL.BACKBONE.NAME](cfg)

# -*- coding: utf-8 -*-

"""
@date: 2020/11/11 下午4:29
@file: build_resnet.py
@author: zj
@description: 
"""

from .resnet_pytorch import ResNet_Pytorch
from .. import registry


@registry.RECOGNIZER.register('ResNet')
def build_resnet(cfg):
    type = cfg.MODEL.RECOGNIZER.NAME

    if type == 'R50_Pytorch':
        return ResNet_Pytorch(cfg)
    else:
        raise ValueError(f'{type} does not exist')

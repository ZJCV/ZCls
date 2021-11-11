# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午2:37
@file: resnet.py
@author: zj
@description: 
"""

from zcls.model import registry
from zcls.model.misc import load_pretrained_weights
from ..base_recognizer import BaseRecognizer

url_map = {
    'resnet18': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/resnet18_imagenet.pth",
    'resnet34': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/resnet34_imagenet.pth",
    'resnet50': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/resnet50_imagenet.pth",
    'resnet101': "",
    'resnet152': "",
    'resnext50_32x4d': "",
    'resnext101_32x8d': ""
}


@registry.RECOGNIZER.register('ResNet')
class ResNet(BaseRecognizer):

    def __init__(self, cfg):
        super().__init__(cfg)

    def _init_weights(self, cfg):
        pretrained_local = cfg.MODEL.RECOGNIZER.PRETRAINED_LOCAL
        pretrained_num_classes = cfg.MODEL.RECOGNIZER.PRETRAINED_NUM_CLASSES
        num_classes = cfg.MODEL.HEAD.NUM_CLASSES
        load_pretrained_weights(self, cfg.MODEL.BACKBONE.ARCH,
                                weights_path=None if pretrained_local == '' else pretrained_local,
                                load_fc=pretrained_num_classes == num_classes,
                                verbose=True,
                                url_map=url_map if cfg.MODEL.RECOGNIZER.PRETRAINED_REMOTE else None
                                )

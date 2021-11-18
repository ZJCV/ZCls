# -*- coding: utf-8 -*-

"""
@date: 2020/12/24 下午7:38
@file: shufflenetv1.py
@author: zj
@description: 
"""

from zcls.model import registry
from zcls.model.misc import load_pretrained_weights
from ..base_recognizer import BaseRecognizer

"""
Note 1: Empirically g = 3 usually has a proper trade-off between accuracy and actual inference time
Note 2: Comparing ShuffleNet 2× with MobileNet whose complexity are comparable (524 vs. 569 MFLOPs)
"""

url_map = {
    'shufflenetv1_3g0_5x': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/shufflenetv1_3g0_5x_imagenet_de2ce242.pth",
    'shufflenetv1_3g1x': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/shufflenetv1_3g1x_imagenet_4b9454f2.pth",
    'shufflenetv1_3g1_5x': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/shufflenetv1_3g1_5x_imagenet_9ac247e1.pth",
    'shufflenetv1_3g2x': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/shufflenetv1_3g2x_imagenet_9835481a.pth",
    'shufflenetv1_8g0_5x': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/shufflenetv1_8g0_5x_imagenet_5940e6d3.pth",
    'shufflenetv1_8g1x': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/shufflenetv1_8g1x_imagenet_941bfdc6.pth",
    'shufflenetv1_8g1_5x': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/shufflenetv1_8g1_5x_imagenet_76b3f38d.pth",
    'shufflenetv1_8g2x': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/shufflenetv1_8g2x_imagenet_9fb71642.pth",
}


@registry.RECOGNIZER.register('ShuffleNetV1')
class ShuffleNetV1(BaseRecognizer):

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

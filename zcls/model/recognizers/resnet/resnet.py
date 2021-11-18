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
    'resnet18': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/resnet18_imagenet_384d0b7e.pth",
    'resnet34': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/resnet34_imagenet_e4448618.pth",
    'resnet50': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/resnet50_imagenet_025f6510.pth",
    'resnet101': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/resnet101_imagenet_cb164cb4.pth",
    'resnet152': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/resnet152_imagenet_a8be90bf.pth",
    'resnext50_32x4d': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/resnext50_32x4d_imagenet_a21b7284.pth",
    'resnext101_32x8d': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/resnext101_32x8d_imagenet_b4025795.pth",
}


@registry.RECOGNIZER.register('ResNet')
class ResNet(BaseRecognizer):

    def __init__(self, cfg):
        super().__init__(cfg)

    def _init_weights(self, cfg):
        pretrained_local = cfg.MODEL.RECOGNIZER.PRETRAINED_LOCAL
        pretrained_num_classes = cfg.MODEL.RECOGNIZER.PRETRAINED_NUM_CLASSES
        num_classes = cfg.MODEL.HEAD.NUM_CLASSES

        model_name = cfg.MODEL.BACKBONE.ARCH
        assert isinstance(model_name, str)
        if model_name.startswith('repvgg'):
            if len(cfg.MODEL.CONV.ADD_BLOCKS) == 0:
                model_name += '_infer'
            else:
                model_name += '_train'
        load_pretrained_weights(self, model_name,
                                weights_path=None if pretrained_local == '' else pretrained_local,
                                load_fc=pretrained_num_classes == num_classes,
                                verbose=True,
                                url_map=url_map if cfg.MODEL.RECOGNIZER.PRETRAINED_REMOTE else None
                                )

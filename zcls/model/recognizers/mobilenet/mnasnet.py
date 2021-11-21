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

url_map = {
    'mnasnet_b1_0_5': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/mnasnet0_5_imagenet_395ba514.pth",
    'mnasnet_b1_1_0': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/mnasnet1_0_imagenet_51c92e88.pth",
}


@registry.RECOGNIZER.register('MNASNet')
class MNASNet(BaseRecognizer):

    def __init__(self, cfg):
        super().__init__(cfg)

    def _init_weights(self, cfg):
        pretrained_local = cfg.MODEL.RECOGNIZER.PRETRAINED_LOCAL
        pretrained_num_classes = cfg.MODEL.RECOGNIZER.PRETRAINED_NUM_CLASSES
        num_classes = cfg.MODEL.HEAD.NUM_CLASSES

        model_name = cfg.MODEL.BACKBONE.ARCH
        assert isinstance(model_name, str)
        if model_name.startswith('mnasnet_b1'):
            if cfg.MODEL.COMPRESSION.WIDTH_MULTIPLIER == 0.5:
                model_name += '_0_5'
            elif cfg.MODEL.COMPRESSION.WIDTH_MULTIPLIER == 1.0:
                model_name += '_1_0'
            else:
                raise ValueError('No supports values')
        load_pretrained_weights(self, model_name,
                                weights_path=None if pretrained_local == '' else pretrained_local,
                                load_fc=pretrained_num_classes == num_classes,
                                verbose=True,
                                url_map=url_map if cfg.MODEL.RECOGNIZER.PRETRAINED_REMOTE else None
                                )

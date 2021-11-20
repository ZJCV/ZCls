# -*- coding: utf-8 -*-

"""
@date: 2021/6/4 下午5:23
@file: ghostnet.py
@author: zj
@description: 
"""

from zcls.model import registry
from zcls.model.misc import load_pretrained_weights
from ..base_recognizer import BaseRecognizer

url_map = {
    'ghostnet_x1_0': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/ghostnet_x1_0_imagenet_0000e4da.pth",
}


@registry.RECOGNIZER.register('GhostNet')
class GhostNet(BaseRecognizer):

    def __init__(self, cfg):
        super().__init__(cfg)

    def _init_weights(self, cfg):
        if cfg.MODEL.COMPRESSION.WIDTH_MULTIPLIER == 1.0:
            model_name = 'ghostnet_x1_0'
        else:
            model_name = 'GhostNet'

        pretrained_local = cfg.MODEL.RECOGNIZER.PRETRAINED_LOCAL
        pretrained_num_classes = cfg.MODEL.RECOGNIZER.PRETRAINED_NUM_CLASSES
        num_classes = cfg.MODEL.HEAD.NUM_CLASSES

        load_pretrained_weights(self, model_name,
                                weights_path=None if pretrained_local == '' else pretrained_local,
                                load_fc=pretrained_num_classes == num_classes,
                                verbose=True,
                                url_map=url_map if cfg.MODEL.RECOGNIZER.PRETRAINED_REMOTE else None
                                )

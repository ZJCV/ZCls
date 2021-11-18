# -*- coding: utf-8 -*-

"""
@date: 2020/12/2 下午9:38
@file: mobilenetv1.py
@author: zj
@description: 
"""

from zcls.model import registry
from zcls.model.misc import load_pretrained_weights
from ..base_recognizer import BaseRecognizer

url_map = {
    'MobileNetV2': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/mobilenet_v2_imagenet_1f638b7a.pth",
}

@registry.RECOGNIZER.register('MobileNetV2')
class MobileNetV2(BaseRecognizer):

    def __init__(self, cfg):
        super().__init__(cfg)

    def _init_weights(self, cfg):
        pretrained_local = cfg.MODEL.RECOGNIZER.PRETRAINED_LOCAL
        pretrained_num_classes = cfg.MODEL.RECOGNIZER.PRETRAINED_NUM_CLASSES
        num_classes = cfg.MODEL.HEAD.NUM_CLASSES

        model_name = cfg.MODEL.BACKBONE.NAME
        assert isinstance(model_name, str)
        load_pretrained_weights(self, model_name,
                                weights_path=None if pretrained_local == '' else pretrained_local,
                                load_fc=pretrained_num_classes == num_classes,
                                verbose=True,
                                url_map=url_map if cfg.MODEL.RECOGNIZER.PRETRAINED_REMOTE else None
                                )
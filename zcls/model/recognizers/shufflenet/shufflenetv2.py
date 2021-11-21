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
G1) Equal channel width minimizes memory access cost (MAC)
G2) Excessive group convolution increases MAC.
G3) Network fragmentation reduces degree of parallelism
G4) Element-wise operations are non-negligible.
"""

url_map = {
    'shufflenet_v2_x0_5': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/shufflenet_v2_x0_5_imagenet_8fd98bb3.pth",
    'shufflenet_v2_x1_0': "",
    'shufflenet_v2_x1_5': "",
    'shufflenet_v2_x2_0': "",
}


@registry.RECOGNIZER.register('ShuffleNetV2')
class ShuffleNetV2(BaseRecognizer):

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

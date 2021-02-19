# -*- coding: utf-8 -*-

"""
@date: 2021/2/19 下午6:51
@file: torchvision_mobilenetv2.py
@author: zj
@description: 
"""

from abc import ABC
import torch.nn as nn
from torchvision.models import mobilenet_v2

from zcls.config.key_word import KEY_OUTPUT
from zcls.model import registry
from zcls.model.norm_helper import freezing_bn, get_norm


class TorchvisionMobileNetV2(nn.Module, ABC):

    def __init__(self,
                 num_classes=1000,
                 width_multiplier=1.0,
                 torchvision_pretrained=False,
                 pretrained_num_classes=1000,
                 fix_bn=False,
                 partial_bn=False,
                 norm_layer=None):
        super(TorchvisionMobileNetV2, self).__init__()
        self.fix_bn = fix_bn
        self.partial_bn = partial_bn

        self.model = mobilenet_v2(pretrained=torchvision_pretrained,
                                  width_mult=width_multiplier,
                                  norm_layer=norm_layer,
                                  num_classes=pretrained_num_classes)

        self.init_weights(num_classes, pretrained_num_classes)

    def init_weights(self, num_classes, pretrained_num_classes):
        if num_classes != pretrained_num_classes:
            fc = self.model.classifier[1]
            fc_features = fc.in_features
            fc = nn.Linear(fc_features, num_classes)

            nn.init.normal_(fc.weight, 0, 0.01)
            nn.init.zeros_(fc.bias)

            self.model.classifier[1] = fc

    def train(self, mode: bool = True):
        super(TorchvisionMobileNetV2, self).train(mode=mode)

        if mode and (self.partial_bn or self.fix_bn):
            freezing_bn(self, partial_bn=self.partial_bn)

        return self

    def forward(self, x):
        x = self.model(x)

        return {KEY_OUTPUT: x}


@registry.RECOGNIZER.register('TorchvisionMobileNetV2')
def build_torchvision_mbv2(cfg):
    torchvision_pretrained = cfg.MODEL.RECOGNIZER.TORCHVISION_PRETRAINED
    pretrained_num_classes = cfg.MODEL.RECOGNIZER.PRETRAINED_NUM_CLASSES
    fix_bn = cfg.MODEL.NORM.FIX_BN
    partial_bn = cfg.MODEL.NORM.PARTIAL_BN
    num_classes = cfg.MODEL.HEAD.NUM_CLASSES
    norm_layer = get_norm(cfg)
    width_multiplier = cfg.MODEL.COMPRESSION.WIDTH_MULTIPLIER

    return TorchvisionMobileNetV2(
        num_classes=num_classes,
        torchvision_pretrained=torchvision_pretrained,
        pretrained_num_classes=pretrained_num_classes,
        width_multiplier=width_multiplier,
        fix_bn=fix_bn,
        partial_bn=partial_bn,
        norm_layer=norm_layer
    )

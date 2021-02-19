# -*- coding: utf-8 -*-

"""
@date: 2021/2/19 下午4:22
@file: torchvision_sfv2.py
@author: zj
@description: 
"""

from abc import ABC

import torch.nn as nn
from torch.nn.modules.module import T
from torchvision.models.shufflenetv2 import shufflenet_v2_x0_5, shufflenet_v2_x1_0, \
    shufflenet_v2_x1_5, shufflenet_v2_x2_0
from zcls.config.key_word import KEY_OUTPUT
from zcls.model import registry

from zcls.model.norm_helper import freezing_bn


class TorchvisionShuffleNetV2(nn.Module, ABC):

    def __init__(self,
                 arch="shufflenet_v2_x2_0",
                 num_classes=1000,
                 torchvision_pretrained=False,
                 pretrained_num_classes=1000,
                 fix_bn=False,
                 partial_bn=False):
        super(TorchvisionShuffleNetV2, self).__init__()

        self.fix_bn = fix_bn
        self.partial_bn = partial_bn

        if arch == 'shufflenet_v2_x2_0':
            self.model = shufflenet_v2_x2_0(pretrained=torchvision_pretrained, num_classes=pretrained_num_classes)
        elif arch == 'shufflenet_v2_x1_5':
            self.model = shufflenet_v2_x1_5(pretrained=torchvision_pretrained, num_classes=pretrained_num_classes)
        elif arch == 'shufflenet_v2_x1_0':
            self.model = shufflenet_v2_x1_0(pretrained=torchvision_pretrained, num_classes=pretrained_num_classes)
        elif arch == 'shufflenet_v2_x0_5':
            self.model = shufflenet_v2_x0_5(pretrained=torchvision_pretrained, num_classes=pretrained_num_classes)
        else:
            raise ValueError('no such value')

        self.init_weights(num_classes, pretrained_num_classes)

    def init_weights(self, num_classes, pretrained_num_classes):
        if num_classes != pretrained_num_classes:
            fc = self.model.fc
            fc_features = fc.in_features
            self.model.fc = nn.Linear(fc_features, num_classes)

            nn.init.normal_(self.model.fc.weight, 0, 0.01)
            nn.init.zeros_(self.model.fc.bias)

    def train(self, mode: bool = True) -> T:
        super(TorchvisionShuffleNetV2, self).train(mode=mode)

        if mode and (self.partial_bn or self.fix_bn):
            freezing_bn(self, partial_bn=self.partial_bn)

        return self

    def forward(self, x):
        x = self.model(x)

        return {KEY_OUTPUT: x}


@registry.RECOGNIZER.register('TorchvisionShuffleNetV2')
def build_torchvision_sfv2(cfg):
    torchvision_pretrained = cfg.MODEL.RECOGNIZER.TORCHVISION_PRETRAINED
    pretrained_num_classes = cfg.MODEL.RECOGNIZER.PRETRAINED_NUM_CLASSES
    num_classes = cfg.MODEL.HEAD.NUM_CLASSES
    arch = cfg.MODEL.BACKBONE.ARCH
    fix_bn = cfg.MODEL.NORM.FIX_BN
    partial_bn = cfg.MODEL.NORM.PARTIAL_BN

    return TorchvisionShuffleNetV2(arch=arch,
                                   num_classes=num_classes,
                                   torchvision_pretrained=torchvision_pretrained,
                                   pretrained_num_classes=pretrained_num_classes,
                                   fix_bn=fix_bn,
                                   partial_bn=partial_bn)

# -*- coding: utf-8 -*-

"""
@date: 2021/2/19 下午8:45
@file: torchvision_mnasnet.py
@author: zj
@description: 
"""

from abc import ABC

import torch.nn as nn
from torchvision.models.mnasnet import mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3

from zcls.config.key_word import KEY_OUTPUT
from zcls.model import registry
from zcls.model.norm_helper import freezing_bn


class TorchvisionMNASNet(nn.Module, ABC):

    def __init__(self,
                 width_multiplier=1.,
                 num_classes=1000,
                 torchvision_pretrained=False,
                 pretrained_num_classes=1000,
                 fix_bn=False,
                 partial_bn=False,
                 ):
        """
        :param width_multiplier: 宽度乘法器
        :param num_classes: 类别数
        :param torchvision_pretrained: 预训练模型
        :param pretrained_num_classes: 假定预训练模型类别数
        :param fix_bn: 固定BN
        :param partial_bn: 仅训练第一层BN
        """
        super(TorchvisionMNASNet, self).__init__()

        self.fix_bn = fix_bn
        self.partial_bn = partial_bn

        if width_multiplier == 0.5:
            self.model = mnasnet0_5(pretrained=torchvision_pretrained, num_classes=pretrained_num_classes)
        elif width_multiplier == 0.75:
            self.model = mnasnet0_75(pretrained=torchvision_pretrained, num_classes=pretrained_num_classes)
        elif width_multiplier == 1.0:
            self.model = mnasnet1_0(pretrained=torchvision_pretrained, num_classes=pretrained_num_classes)
        elif width_multiplier == 1.3:
            self.model = mnasnet1_3(pretrained=torchvision_pretrained, num_classes=pretrained_num_classes)
        else:
            raise ValueError('no such value')

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
        super(TorchvisionMNASNet, self).train(mode=mode)

        if mode and (self.partial_bn or self.fix_bn):
            freezing_bn(self, partial_bn=self.partial_bn)

        return self

    def forward(self, x):
        x = self.model(x)

        return {KEY_OUTPUT: x}


@registry.RECOGNIZER.register('TorchvisionMNASNet')
def build_torchvision_mnasnet(cfg):
    torchvision_pretrained = cfg.MODEL.RECOGNIZER.TORCHVISION_PRETRAINED
    pretrained_num_classes = cfg.MODEL.RECOGNIZER.PRETRAINED_NUM_CLASSES
    num_classes = cfg.MODEL.HEAD.NUM_CLASSES
    # bn
    fix_bn = cfg.MODEL.NORM.FIX_BN
    partial_bn = cfg.MODEL.NORM.PARTIAL_BN
    # compression
    width_multiplier = cfg.MODEL.COMPRESSION.WIDTH_MULTIPLIER

    return TorchvisionMNASNet(
        width_multiplier=width_multiplier,
        num_classes=num_classes,
        torchvision_pretrained=torchvision_pretrained,
        pretrained_num_classes=pretrained_num_classes,
        fix_bn=fix_bn,
        partial_bn=partial_bn
    )

# -*- coding: utf-8 -*-

"""
@date: 2021/2/20 上午10:28
@file: torchvision_resnet.py
@author: zj
@description: 
"""

from abc import ABC

import torch.nn as nn
from torch.nn.modules.module import T
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, \
    resnext101_32x8d

from zcls.config.key_word import KEY_OUTPUT
from zcls.model import registry
from zcls.model.norm_helper import freezing_bn


class TorchvisionResNet(nn.Module, ABC):

    def __init__(self,
                 arch="resnet18",
                 num_classes=1000,
                 torchvision_pretrained=False,
                 pretrained_num_classes=1000,
                 fix_bn=False,
                 partial_bn=False,
                 zero_init_residual=False):
        super(TorchvisionResNet, self).__init__()

        self.num_classes = num_classes
        self.fix_bn = fix_bn
        self.partial_bn = partial_bn

        if arch == 'resnet18':
            self.model = resnet18(pretrained=torchvision_pretrained, num_classes=pretrained_num_classes,
                                  zero_init_residual=zero_init_residual)
        elif arch == 'resnet34':
            self.model = resnet34(pretrained=torchvision_pretrained, num_classes=pretrained_num_classes,
                                  zero_init_residual=zero_init_residual)
        elif arch == 'resnet50':
            self.model = resnet50(pretrained=torchvision_pretrained, num_classes=pretrained_num_classes,
                                  zero_init_residual=zero_init_residual)
        elif arch == 'resnet101':
            self.model = resnet101(pretrained=torchvision_pretrained, num_classes=pretrained_num_classes,
                                   zero_init_residual=zero_init_residual)
        elif arch == 'resnet152':
            self.model = resnet152(pretrained=torchvision_pretrained, num_classes=pretrained_num_classes,
                                   zero_init_residual=zero_init_residual)
        elif arch == 'resnext50_32x4d':
            self.model = resnext50_32x4d(pretrained=torchvision_pretrained, num_classes=pretrained_num_classes,
                                         zero_init_residual=zero_init_residual)
        elif arch == 'resnext101_32x8d':
            self.model = resnext101_32x8d(pretrained=torchvision_pretrained, num_classes=pretrained_num_classes,
                                          zero_init_residual=zero_init_residual)
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
        super(TorchvisionResNet, self).train(mode=mode)

        if mode and (self.partial_bn or self.fix_bn):
            freezing_bn(self, partial_bn=self.partial_bn)

        return self

    def forward(self, x):
        x = self.model(x)

        return {KEY_OUTPUT: x}


@registry.RECOGNIZER.register('TorchvisionResNet')
def build_torchvision_resnet(cfg):
    torchvision_pretrained = cfg.MODEL.RECOGNIZER.TORCHVISION_PRETRAINED
    pretrained_num_classes = cfg.MODEL.RECOGNIZER.PRETRAINED_NUM_CLASSES
    fix_bn = cfg.MODEL.NORM.FIX_BN
    partial_bn = cfg.MODEL.NORM.PARTIAL_BN
    # for backbone
    arch = cfg.MODEL.BACKBONE.ARCH
    zero_init_residual = cfg.MODEL.RECOGNIZER.ZERO_INIT_RESIDUAL
    num_classes = cfg.MODEL.HEAD.NUM_CLASSES

    return TorchvisionResNet(
        arch=arch,
        num_classes=num_classes,
        torchvision_pretrained=torchvision_pretrained,
        pretrained_num_classes=pretrained_num_classes,
        fix_bn=fix_bn,
        partial_bn=partial_bn,
        zero_init_residual=zero_init_residual
    )

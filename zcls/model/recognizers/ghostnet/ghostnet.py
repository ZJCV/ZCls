# -*- coding: utf-8 -*-

"""
@date: 2021/6/4 下午5:23
@file: ghostnet.py
@author: zj
@description: 
"""

import torch.nn as nn

from zcls.model import registry
from ..base_recognizer import BaseRecognizer


@registry.RECOGNIZER.register('GhostNet')
class GhostNet(BaseRecognizer):

    def __init__(self, cfg):
        super().__init__(cfg)

    def init_weights(self, pretrained, pretrained_num_classes, num_classes):
        # Using super class method to load pretraining weights
        super(GhostNet, self).init_weights(pretrained, pretrained_num_classes, pretrained_num_classes)
        if num_classes != pretrained_num_classes:
            in_channels = self.head.conv2.in_channels
            conv2 = nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

            nn.init.kaiming_normal_(conv2.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(conv2.bias)

            self.head.conv2 = conv2

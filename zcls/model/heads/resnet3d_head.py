# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午4:09
@file: resnet_head.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn


class ResNet3DHead(nn.Module):

    def __init__(self, feature_dims, num_classes):
        super(ResNet3DHead, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1, 1))
        self.fc = nn.Linear(feature_dims, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

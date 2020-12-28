# -*- coding: utf-8 -*-

"""
@date: 2020/12/4 下午3:44
@file: mobilenetv2_head.py
@author: zj
@description: 
"""
from abc import ABC

import torch
import torch.nn as nn


class ShuffleNetV2Head(nn.Module, ABC):

    def __init__(self,
                 # 输入特征维度
                 feature_dims=1024,
                 # 类别数
                 num_classes=1000,
                 ):
        super(ShuffleNetV2Head, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
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

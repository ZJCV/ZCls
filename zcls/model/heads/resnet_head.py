# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午4:09
@file: resnet_head.py
@author: zj
@description: 
"""
from abc import ABC

import torch
import torch.nn as nn


class ResNetHead(nn.Module, ABC):

    def __init__(self,
                 # 输入特征维度
                 feature_dims=2048,
                 # 类别数
                 num_classes=1000
                 ):
        super(ResNetHead, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(feature_dims, num_classes)

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

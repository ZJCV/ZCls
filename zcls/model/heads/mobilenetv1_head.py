# -*- coding: utf-8 -*-

"""
@date: 2020/12/3 下午8:17
@file: mobilenetv1_head.py
@author: zj
@description: 
"""
from abc import ABC

import torch
import torch.nn as nn


class MobileNetV1Head(nn.Module, ABC):

    def __init__(self,
                 # 输入特征维度
                 feature_dims=1024,
                 # 类别数
                 num_classes=1000,
                 # 随机失活概率
                 dropout_rate=0.
                 ):
        super(MobileNetV1Head, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate, inplace=True)
        self.fc = nn.Linear(feature_dims, num_classes)

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

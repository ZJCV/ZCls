# -*- coding: utf-8 -*-

"""
@date: 2020/12/30 下午6:48
@file: general_head_2d.py
@author: zj
@description: 
"""
from abc import ABC

import torch
import torch.nn as nn


class MobileNetV3Head(nn.Module, ABC):

    def __init__(self,
                 # 输入特征维度
                 feature_dims=960,
                 # 中间特征维度
                 inner_dims=1280,
                 # 类别数
                 num_classes=1000,
                 # 卷积层类型
                 conv_layer=None,
                 # 激活层类型
                 act_layer=None
                 ):
        super(MobileNetV3Head, self).__init__()

        if conv_layer is None:
            conv_layer = nn.Conv2d
        if act_layer is None:
            act_layer = nn.Hardswish

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = conv_layer(feature_dims, inner_dims, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = conv_layer(inner_dims, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.act = act_layer()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = torch.flatten(x, 1)
        return x

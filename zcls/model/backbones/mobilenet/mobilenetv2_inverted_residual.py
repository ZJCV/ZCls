# -*- coding: utf-8 -*-

"""
@date: 2020/12/4 上午11:39
@file: mobilenetv2_inverted_residual.py
@author: zj
@description: MobileNetV2 反向残差块
"""
from abc import ABC

import torch.nn as nn


class MobileNetV2InvertedResidual(nn.Module, ABC):

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion_rate=1,
                 stride=1,
                 padding=1,
                 conv_layer=None,
                 norm_layer=None,
                 act_layer=None,
                 ):
        """
        MobileNetV2的反向残差块由一个膨胀卷积和一个深度可分离卷积(depth-wise conv + point-wise conv)组成
        参考torchvision实现:
        1. 当膨胀率大于1时，执行膨胀卷积操作；
        2. 当深度卷积步长为1且输入/输出通道数相同时，执行残差连接
        3. 反向残差块的最后不执行激活操作
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param expansion_rate: 膨胀因子
        :param stride: 卷积层步长
        :param padding: 卷积层零填充
        :param conv_layer: 卷积层类型
        :param norm_layer: 归一化层类型
        :param act_layer: 激活层类型
        """
        super(MobileNetV2InvertedResidual, self).__init__()

        if conv_layer is None:
            conv_layer = nn.Conv2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU6

        # 计算隐藏层输入通道数
        hidden_channels = int(expansion_rate * in_channels)
        features = list()
        if expansion_rate != 1:
            features.append(nn.Sequential(
                conv_layer(in_channels, hidden_channels, kernel_size=1, stride=1, bias=False),
                norm_layer(hidden_channels),
                act_layer(inplace=True)
            ))

        # 深度卷积
        features.append(nn.Sequential(
            conv_layer(hidden_channels, hidden_channels, kernel_size=3, stride=stride, padding=padding, bias=False,
                       groups=hidden_channels),
            norm_layer(hidden_channels),
            act_layer(inplace=True)
        ))

        # 逐点卷积
        features.append(nn.Sequential(
            conv_layer(hidden_channels, out_channels, kernel_size=1, stride=1, bias=False),
            norm_layer(out_channels)
        ))

        self.conv = nn.Sequential(*features)
        self.use_res_connect = stride == 1 and in_channels == out_channels

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

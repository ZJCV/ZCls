# -*- coding: utf-8 -*-

"""
@date: 2020/12/2 下午9:46
@file: mobilenetv1_block.py
@author: zj
@description: 
"""
from abc import ABC

import torch.nn as nn


class MobileNetV1Block(nn.Module, ABC):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 padding=1,
                 conv_layer=None,
                 norm_layer=None,
                 act_layer=None,
                 ) -> None:
        """
        Block = depth-wise convolution + point-wise convolution
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param stride: 卷积层步长
        :param padding: 卷积层零填充
        :param conv_layer: 卷积层类型
        :param norm_layer: 归一化层类型
        :param act_layer: 激活层类型
        """
        super(MobileNetV1Block, self).__init__()

        if conv_layer is None:
            conv_layer = nn.Conv2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU

        self.conv1 = conv_layer(in_channels, in_channels, kernel_size=3, stride=stride, padding=padding,
                                groups=in_channels, bias=False)
        self.bn1 = norm_layer(in_channels)
        self.relu = act_layer(inplace=True)

        self.conv2 = conv_layer(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn2 = norm_layer(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

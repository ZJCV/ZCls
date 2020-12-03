# -*- coding: utf-8 -*-

"""
@date: 2020/12/2 下午9:46
@file: mobilenetv1_block.py
@author: zj
@description: 
"""

import torch.nn as nn


class MobileNetV1Block(nn.Module):
    """
    每个Block由一个逐通道卷积（depthwise convolution）和一个逐点卷积（pointwise convolution）组成
    其输出通道数保持不变，由逐通道卷积决定是否进行空间下采样
    """

    def __init__(self,
                 # 输入通道数
                 inplanes,
                 # 输出通道数
                 planes,
                 # 卷积层步长
                 stride=1,
                 # 卷积层零填充
                 padding=1,
                 # 卷积层类型
                 conv_layer=None,
                 # 归一化层类型
                 norm_layer=None,
                 # 激活层类型
                 act_layer=None,
                 ) -> None:
        super(MobileNetV1Block, self).__init__()

        if conv_layer is None:
            conv_layer = nn.Conv2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU

        self.conv_layer = conv_layer
        self.norm_layer = norm_layer
        self.act_layer = act_layer

        self.conv1 = self.conv_layer(inplanes, inplanes, kernel_size=3, stride=stride, padding=padding,
                                     groups=inplanes, bias=False)
        self.bn = self.norm_layer(inplanes)
        self.relu = self.act_layer(inplace=True)

        self.conv2 = self.conv_layer(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn2 = self.norm_layer(planes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

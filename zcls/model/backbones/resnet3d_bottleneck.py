# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午3:15
@file: bottleneck.py
@author: zj
@description: 
"""

import torch.nn as nn


class ResNet3DBottleneck(nn.Module):
    """
    依次执行大小为Tx1x1、Tx3x3、1x1x1的卷积操作，
    如果进行下采样，那么使用第二个卷积层对输入空间尺寸进行减半操作
    分两种膨胀类型：3x1x1和3x3x3
    """
    expansion = 4

    def __init__(self,
                 # 输入通道数
                 inplanes,
                 # 输出通道数
                 planes,
                 # 空间步长
                 spatial_stride=1,
                 # 时间步长
                 temporal_stride=1,
                 # 下采样
                 downsample=None,
                 # 是否膨胀
                 inflate=False,
                 # 膨胀类型，作用于Bottleneck
                 inflate_style='3x1x1',
                 # 卷积层类型
                 conv_layer=None,
                 # 归一化层类型
                 norm_layer=None,
                 # 激活层类型
                 act_layer=None
                 ):
        super(ResNet3DBottleneck, self).__init__()
        assert inflate_style in ("3x1x1", "3x3x3")

        if conv_layer is None:
            conv_layer = nn.Conv3d
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if act_layer is None:
            act_layer = nn.ReLU

        if inflate:
            if inflate_style == '3x1x1':
                conv1_kernel_size = (3, 1, 1)
                conv1_stride = (temporal_stride, 1, 1)
                conv1_padding = (1, 0, 0)
                conv2_kernel_size = (1, 3, 3)
                conv2_stride = (1, spatial_stride, spatial_stride)
                conv2_padding = (0, 1, 1)
            else:
                conv1_kernel_size = (1, 1, 1)
                conv1_stride = (1, 1, 1)
                conv1_padding = (0, 0, 0)
                conv2_kernel_size = (3, 3, 3)
                conv2_stride = (temporal_stride, spatial_stride, spatial_stride)
                conv2_padding = (1, 1, 1)
        else:
            conv1_kernel_size = (1, 1, 1)
            conv1_stride = (1, 1, 1)
            conv1_padding = (0, 0, 0)
            conv2_kernel_size = (1, 3, 3)
            conv2_stride = (1, spatial_stride, spatial_stride)
            conv2_padding = (0, 1, 1)

        self.downsample = downsample

        self.conv1 = conv_layer(inplanes, planes, kernel_size=conv1_kernel_size,
                                stride=conv1_stride, padding=conv1_padding, bias=False)
        self.bn1 = norm_layer(planes)

        self.conv2 = conv_layer(planes, planes, kernel_size=conv2_kernel_size,
                                stride=conv2_stride, padding=conv2_padding, bias=False)
        self.bn2 = norm_layer(planes)

        self.conv3 = conv_layer(planes, planes * self.expansion, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = act_layer(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

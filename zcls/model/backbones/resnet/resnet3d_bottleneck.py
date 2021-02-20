# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午3:15
@file: bottleneck.py
@author: zj
@description: 
"""
from abc import ABC

import torch.nn as nn


class ResNet3DBottleneck(nn.Module, ABC):
    expansion = 4

    def __init__(self,
                 in_planes,
                 out_planes,
                 spatial_stride=1,
                 temporal_stride=1,
                 down_sample=None,
                 inflate=False,
                 inflate_style='3x1x1',
                 groups=1,
                 base_width=64,
                 conv_layer=None,
                 norm_layer=None,
                 act_layer=None
                 ):
        """
        依次执行大小为Tx1x1、Tx3x3、1x1x1的卷积操作，
        如果进行下采样，那么使用第二个卷积层对输入空间尺寸进行减半操作
        分两种膨胀类型：3x1x1和3x3x3
        :param in_planes: 输入通道数
        :param out_planes: 输出通道数
        :param spatial_stride: 空间步长
        :param temporal_stride: 时间步长
        :param down_sample: 下采样
        :param inflate: 是否膨胀
        :param inflate_style: 膨胀类型，作用于Bottleneck
        :param groups: cardinality
        :param base_width: 基础宽度
        :param conv_layer: 卷积层类型
        :param norm_layer: 归一化层类型
        :param act_layer: 激活层类型
        """
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

        self.down_sample = down_sample

        width = int(out_planes * (base_width / 64.)) * groups
        self.conv1 = conv_layer(in_planes, width, kernel_size=conv1_kernel_size,
                                stride=conv1_stride, padding=conv1_padding, bias=False)
        self.bn1 = norm_layer(width)

        self.conv2 = conv_layer(width, width, kernel_size=conv2_kernel_size,
                                stride=conv2_stride, padding=conv2_padding, bias=False,
                                groups=groups)
        self.bn2 = norm_layer(width)

        self.conv3 = conv_layer(width, out_planes * self.expansion, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.bn3 = norm_layer(out_planes * self.expansion)

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

        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out

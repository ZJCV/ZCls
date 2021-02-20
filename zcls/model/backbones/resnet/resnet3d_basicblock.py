# -*- coding: utf-8 -*-

"""
@date: 2020/12/8 下午7:58
@file: resnet3d_basicblock.py
@author: zj
@description: 
"""
from abc import ABC

import torch.nn as nn


class ResNet3DBasicBlock(nn.Module, ABC):
    expansion = 1

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
        使用两个Tx3x3卷积，如果进行下采样，那么使用第一个卷积层对输入时空尺寸进行减半操作
        如果执行膨胀操作，仅作用于第一个卷积层，第二个卷积层的kernel_size大小为1x3x3
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
        super(ResNet3DBasicBlock, self).__init__()
        assert inflate_style in ("3x1x1", "3x3x3")
        if groups != 1 or base_width != 64:
            raise ValueError('ResNet3DBasicBlock only supports groups=1 and base_width=64')

        if conv_layer is None:
            conv_layer = nn.Conv3d
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if act_layer is None:
            act_layer = nn.ReLU

        if inflate:
            conv1_kernel_size = (3, 3, 3)
            conv1_stride = (temporal_stride, spatial_stride, spatial_stride)
            conv1_padding = (1, 1, 1)
        else:
            conv1_kernel_size = (1, 3, 3)
            conv1_stride = (1, spatial_stride, spatial_stride)
            conv1_padding = (0, 1, 1)

        self.down_sample = down_sample

        self.conv1 = conv_layer(in_planes, out_planes, kernel_size=conv1_kernel_size,
                                stride=conv1_stride, padding=conv1_padding,
                                bias=False)
        self.bn1 = norm_layer(out_planes)

        self.conv2 = conv_layer(out_planes, out_planes, kernel_size=(1, 3, 3),
                                stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
        self.bn2 = norm_layer(out_planes)

        self.relu = act_layer(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out

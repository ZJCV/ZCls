# -*- coding: utf-8 -*-

"""
@date: 2020/12/8 下午7:58
@file: resnet3d_basicblock.py
@author: zj
@description: 
"""

import torch.nn as nn


class ResNet3DBasicBlock(nn.Module):
    """
    使用两个Tx3x3卷积，如果进行下采样，那么使用第一个卷积层对输入时空尺寸进行减半操作
    如果执行膨胀操作，仅作用于第一个卷积层，第二个卷积层的kernel_size大小为1x3x3
    """
    expansion = 1

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
        super(ResNet3DBasicBlock, self).__init__()
        assert inflate_style in ("3x1x1", "3x3x3")

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

        self.downsample = downsample

        self.conv1 = conv_layer(inplanes, planes, kernel_size=conv1_kernel_size,
                                stride=conv1_stride, padding=conv1_padding,
                                bias=False)
        self.bn1 = norm_layer(planes)

        self.conv2 = conv_layer(planes, planes, kernel_size=(1, 3, 3),
                                stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
        self.bn2 = norm_layer(planes)

        self.relu = act_layer(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

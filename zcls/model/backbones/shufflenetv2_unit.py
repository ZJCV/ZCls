# -*- coding: utf-8 -*-

"""
@date: 2020/12/24 下午8:04
@file: shufflenetv1_unit.py
@author: zj
@description: 
"""
from abc import ABC

import torch

import torch.nn as nn


def channel_shuffle(x, groups):
    """
    # >>> a = torch.arange(12)
    # >>> b = a.reshape(3,4)
    # >>> c = b.transpose(1,0).contiguous()
    # >>> d = c.view(3,4)
    # >>> a
    # tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
    # >>> b
    # tensor([[ 0,  1,  2,  3],
    #         [ 4,  5,  6,  7],
    #         [ 8,  9, 10, 11]])
    # >>> c
    # tensor([[ 0,  4,  8],
    #         [ 1,  5,  9],
    #         [ 2,  6, 10],
    #         [ 3,  7, 11]])
    # >>> d
    # tensor([[ 0,  4,  8,  1],
    #         [ 5,  9,  2,  6],
    #         [10,  3,  7, 11]])
    """
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class ShuffleNetV2Unit(nn.Module, ABC):
    """
    通道分块 + MobileNetV2的反向残差块（由一个膨胀卷积和一个深度可分离卷积组成）+ 通道重排（shortcut path + residual path）
    """

    def __init__(self,
                 # 输入通道
                 in_planes,
                 # 输出通道
                 out_planes,
                 # 步长
                 stride,
                 # 作用于shortcut path
                 down_sample=None,
                 # 卷积层类型
                 conv_layer=None,
                 # 归一化层类型
                 norm_layer=None,
                 # 激活层类型
                 act_layer=None,
                 ):
        super().__init__()

        if conv_layer is None:
            conv_layer = nn.Conv2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU

        self.conv1 = conv_layer(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm1 = norm_layer(out_planes)

        self.conv2 = conv_layer(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False,
                                     groups=out_planes)
        self.norm2 = norm_layer(out_planes)

        self.conv3 = conv_layer(out_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm3 = norm_layer(out_planes)

        self.act = act_layer(inplace=True)
        self.stride = stride
        self.down_sample = down_sample

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
        else:
            x1 = x
            x2 = x

        out = self.conv1(x1)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = self.conv3(out)
        out = self.norm3(out)
        out = self.act(out)

        if self.down_sample is not None:
            x2 = self.down_sample(x2)

        out = torch.cat((out, x2), dim=1)

        out = channel_shuffle(out, 2)
        return out

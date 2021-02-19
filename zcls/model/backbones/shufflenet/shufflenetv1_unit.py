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

from ..misc import channel_shuffle


class ShuffleNetV1Unit(nn.Module, ABC):

    def __init__(self,
                 in_channels,
                 out_channels,
                 groups,
                 stride,
                 downsample=None,
                 with_group=True,
                 conv_layer=None,
                 norm_layer=None,
                 act_layer=None,
                 ):
        """
        when stride=1, Unit = Add(Input, GConv(DWConv(Channel Shuffle(GConv(Input)))));
        when stride=2, Unit = Concat(AvgPool(Input), GConv(DWConv(Channel Shuffle(GConv(Input)))))
        :param in_channels: 输入通道
        :param out_channels: 输出通道
        :param groups: 分组数
        :param stride: 步长
        :param downsample: 作用于shortcut path
        :param with_group: 是否对第一个1x1逐点卷积应用分组
        :param conv_layer: 卷积层类型
        :param norm_layer: 归一化层类型
        :param act_layer: 激活层类型
        """
        super().__init__()
        assert out_channels % groups == 0

        if conv_layer is None:
            conv_layer = nn.Conv2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU

        out_channels = out_channels if stride == 1 else out_channels - in_channels
        groups = groups if with_group else 1
        self.conv1 = conv_layer(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False, groups=groups)
        self.norm1 = norm_layer(out_channels)

        self.conv2 = conv_layer(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False,
                                groups=out_channels)
        self.norm2 = norm_layer(out_channels)

        self.conv3 = conv_layer(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False,
                                groups=groups)
        self.norm3 = norm_layer(out_channels)

        self.act = act_layer(inplace=True)
        self.down_sample = downsample
        self.groups = groups
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = channel_shuffle(out, groups=self.groups)

        out = self.conv2(out)
        out = self.norm2(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        if self.stride == 2:
            out = torch.cat((out, identity), dim=1)
        else:
            out = out + identity
        out = self.act(out)
        return out

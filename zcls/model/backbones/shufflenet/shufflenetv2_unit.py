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


class ShuffleNetV2Unit(nn.Module, ABC):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 downsample=None,
                 conv_layer=None,
                 norm_layer=None,
                 act_layer=None,
                 ):
        """
        refer to Torchvision realization: https://github.com/pytorch/vision/blob/master/torchvision/models/shufflenetv2.py
        when stride = 1, Unit = Channel Shuffle(Concat(Channel Split(Input), Conv(DWConv(Conv(Channel Split(Input))))));
        when stride = 2, Unit = Channel Shuffle(Concat(Conv(DWConv(Input)), Conv(DWConv(Conv(Input)))));
        official realization (https://github.com/megvii-model/ShuffleNet-Series/blob/master/ShuffleNetV2/blocks.py) has
         a more ingenious implementation, see [A mismatch in shufflenetv1 about ReLU #53](https://github.com/megvii-model/ShuffleNet-Series/issues/53)
        :param in_channels: 输入通道
        :param out_channels: 输出通道
        :param stride: 步长
        :param downsample: 作用于shortcut path
        :param conv_layer: 卷积层类型
        :param norm_layer: 归一化层类型
        :param act_layer: 激活层类型
        """
        super().__init__()

        if conv_layer is None:
            conv_layer = nn.Conv2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU

        mid_channels = out_channels // 2
        self.conv1 = conv_layer(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm1 = norm_layer(mid_channels)

        self.conv2 = conv_layer(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False,
                                groups=mid_channels)
        self.norm2 = norm_layer(mid_channels)

        self.conv3 = conv_layer(mid_channels, mid_channels, kernel_size=1, stride=1, padding=0,
                                bias=False)
        self.norm3 = norm_layer(mid_channels)

        self.act = act_layer(inplace=True)
        self.stride = stride
        self.down_sample = downsample

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
        else:
            x1 = x
            x2 = x

        out = self.conv1(x2)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = self.conv3(out)
        out = self.norm3(out)
        out = self.act(out)

        if self.down_sample is not None:
            x1 = self.down_sample(x1)

        out = torch.cat((x1, out), dim=1)

        out = channel_shuffle(out, 2)
        return out

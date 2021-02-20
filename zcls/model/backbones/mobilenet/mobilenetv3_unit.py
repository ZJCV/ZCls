# -*- coding: utf-8 -*-

"""
@date: 2020/12/30 下午4:28
@file: mnasnet_unit.py
@author: zj
@description:
"""
from abc import ABC

import torch.nn as nn

from zcls.model.attention_helper import make_attention_block
from zcls.model.layers.hard_swish_wrapper import HardswishWrapper

# Paper suggests 0.9997 momentum, for TensorFlow. Equivalent PyTorch momentum is
# 1.0 - tensorflow.
BN_MOMENTUM = 1 - 0.9997


class MobileNetV3Uint(nn.Module, ABC):

    def __init__(self,
                 in_channels,
                 inner_channels,
                 out_channels,
                 stride=1,
                 kernel_size=3,
                 with_attention=True,
                 reduction=4,
                 attention_type='SqueezeAndExcitationBlock2D',
                 conv_layer=None,
                 norm_layer=None,
                 act_layer=None,
                 sigmoid_type=None
                 ):
        """
        MobileNetV2的反向残差构建块 + Squeeze-and-Excite + h-swish
        :param in_channels: 输入通道数
        :param inner_channels: 中间膨胀块大小
        :param out_channels: 输出通道数
        :param stride: 步长
        :param kernel_size: 卷积核大小
        :param with_attention: 是否使用注意力模块
        :param reduction: 衰减率
        :param attention_type: 注意力模块类型
        :param conv_layer: 卷积层类型
        :param norm_layer: 归一化层类型
        :param act_layer: 激活层类型
        :param sigmoid_type: sigmoid类型
        """
        super(MobileNetV3Uint, self).__init__()

        if conv_layer is None:
            conv_layer = nn.Conv2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = HardswishWrapper
        if sigmoid_type is None:
            sigmoid_type = 'HSigmoid'

        self.with_attention = with_attention

        self.expansion = nn.Sequential(
            conv_layer(in_channels, inner_channels, kernel_size=1, stride=1, padding=0, bias=False),
            norm_layer(inner_channels, momentum=BN_MOMENTUM),
            act_layer(inplace=True)
        )

        if kernel_size == 3:
            padding = 1
        elif kernel_size == 5:
            padding = 2
        else:
            padding = 0
        self.conv1 = conv_layer(inner_channels, inner_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=False, groups=inner_channels)
        self.norm1 = norm_layer(inner_channels, momentum=BN_MOMENTUM)
        self.act = act_layer(inplace=True)

        self.conv2 = conv_layer(inner_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm2 = norm_layer(out_channels, momentum=BN_MOMENTUM)

        if self.with_attention:
            self.attention = make_attention_block(inner_channels, reduction, attention_type, sigmoid_type=sigmoid_type)

        self.apply_residual = (in_channels == out_channels and stride == 1)

    def forward(self, x):
        identity = x

        out = self.expansion(x)

        out = self.conv1(out)
        out = self.norm1(out)
        out = self.act(out)

        if self.with_attention:
            out = self.attention(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.apply_residual:
            out = out + identity
        # Linear pointwise. Note that there's no activation.
        # out = self.act(out)
        return out

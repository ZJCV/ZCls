# -*- coding: utf-8 -*-

"""
@date: 2020/12/30 下午4:28
@file: mnasnet_unit.py
@author: zj
@description:
"""
from abc import ABC

import torch.nn as nn

from ..attention_helper import make_attention_block
from ..layers.hard_swish_wrapper import HardswishWrapper

# Paper suggests 0.9997 momentum, for TensorFlow. Equivalent PyTorch momentum is
# 1.0 - tensorflow.
BN_MOMENTUM = 1 - 0.9997


class MobileNetV3Uint(nn.Module, ABC):
    """
    MobileNetV2的反向残差构建块 + Squeeze-and-Excite + h-swish
    """

    def __init__(self,
                 # 输入通道数
                 in_planes,
                 # 中间膨胀块大小
                 inner_planes,
                 # 输出通道数
                 out_planes,
                 # 步长
                 stride=1,
                 # 卷积核大小
                 kernel_size=3,
                 # 是否使用注意力模块
                 with_attention=True,
                 # 衰减率
                 reduction=4,
                 # 注意力模块类型
                 attention_type='SqueezeAndExcitationBlock2D',
                 # 卷积层类型
                 conv_layer=None,
                 # 归一化层类型
                 norm_layer=None,
                 # 激活层类型
                 act_layer=None
                 ):
        super(MobileNetV3Uint, self).__init__()

        if conv_layer is None:
            conv_layer = nn.Conv2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = HardswishWrapper

        self.with_attention = with_attention

        self.expansion = nn.Sequential(
            conv_layer(in_planes, inner_planes, kernel_size=1, stride=1, padding=0, bias=False),
            norm_layer(inner_planes, momentum=BN_MOMENTUM),
            act_layer(inplace=True)
        )

        if kernel_size == 3:
            padding = 1
        elif kernel_size == 5:
            padding = 2
        else:
            padding = 0
        self.conv1 = conv_layer(inner_planes, inner_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=False, groups=inner_planes)
        self.norm1 = norm_layer(inner_planes, momentum=BN_MOMENTUM)
        self.act = act_layer(inplace=True)

        self.conv2 = conv_layer(inner_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm2 = norm_layer(out_planes, momentum=BN_MOMENTUM)

        if self.with_attention:
            self.attention = make_attention_block(inner_planes, reduction, attention_type)

        if stride > 1:
            # AvgPool + Conv1x1
            self.down_sample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                conv_layer(in_planes, out_planes, kernel_size=1, stride=1, bias=False),
                norm_layer(out_planes),
            )
        elif in_planes != out_planes:
            # Conv1x1
            self.down_sample = nn.Sequential(
                conv_layer(in_planes, out_planes, kernel_size=1, stride=1, bias=False),
                norm_layer(out_planes),
            )
        else:
            self.down_sample = None

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

        if self.down_sample is not None:
            identity = self.down_sample(identity)

        out = out + identity
        # Linear pointwise. Note that there's no activation.
        # out = self.act(out)
        return out

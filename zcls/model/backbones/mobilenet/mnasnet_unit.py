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

# Paper suggests 0.9997 momentum, for TensorFlow. Equivalent PyTorch momentum is
# 1.0 - tensorflow.
BN_MOMENTUM = 1 - 0.9997


class MNASNetUint(nn.Module, ABC):

    def __init__(self,
                 in_planes,
                 out_planes,
                 stride=1,
                 kernel_size=3,
                 expansion_rate=1,
                 with_attention=True,
                 reduction=4,
                 attention_type='SqueezeAndExcitationBlock2D',
                 conv_layer=None,
                 norm_layer=None,
                 act_layer=None
                 ):
        """
        整合了MBConv3(3x3/5x5/SE)、MBConv6(3x3/5x5/SE)、SepConv(k3x3)实现
        :param in_planes: 输入通道数
        :param out_planes: 输出通道数
        :param stride: 步长
        :param kernel_size: 卷积核大小
        :param expansion_rate: 膨胀因子，适用于mbv2的反向残差块
        :param with_attention: 是否使用注意力模块
        :param reduction: 衰减率
        :param attention_type: 注意力模块类型
        :param conv_layer: 卷积层类型
        :param norm_layer: 归一化层类型
        :param act_layer: 激活层类型
        """
        super(MNASNetUint, self).__init__()
        assert isinstance(expansion_rate, int) and expansion_rate >= 1

        if conv_layer is None:
            conv_layer = nn.Conv2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU

        # if expansion_rate, use MBConv6(k3x3)/MBConv3(k5x5), else use SepConv(k3x3)
        self.with_expansion = expansion_rate > 1
        # if with attention, use MBConv3(k5x5)
        self.with_attention = with_attention

        inner_planes = in_planes * expansion_rate
        if self.with_expansion:
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

        self.apply_residual = (in_planes == out_planes and stride == 1)

    def forward(self, x):
        identity = x

        if self.with_expansion:
            x = self.expansion(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        if self.with_attention:
            out = self.attention(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.apply_residual:
            out = out + identity
        # 参考Torchvision.
        # Linear pointwise. Note that there's no activation.
        # out = self.act(out)

        return out

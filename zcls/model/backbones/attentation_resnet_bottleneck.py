# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午3:15
@file: bottleneck.py
@author: zj
@description: 
"""

import torch.nn as nn

from zcls.model.layers.global_context_block import GlobalContextBlock2D
from zcls.model.layers.squeeze_and_excitation_block import SqueezeAndExcitationBlock2D
from zcls.model.layers.non_local_embedded_gaussian import NonLocal2DEmbeddedGaussian
from zcls.model.layers.simplified_non_local_embedded_gaussian import SimplifiedNonLocal2DEmbeddedGaussian


class AttentionResNetBottleneck(nn.Module):
    """
    依次执行大小为1x1、3x3、1x1的卷积操作，如果进行下采样，那么使用第二个卷积层对输入空间尺寸进行减半操作
    标准SE操作：在残差块操作后（after1x1）嵌入SE操作；
    """
    expansion = 4

    def __init__(self,
                 # 输入通道数
                 inplanes,
                 # 输出通道数
                 planes,
                 # 步长
                 stride=1,
                 # 下采样
                 downsample=None,
                 # 是否使用注意力模块
                 with_attention=True,
                 # 衰减率
                 reduction=16,
                 # 注意力模块类型
                 attention_type='GlobalContextBlock2D',
                 # cardinality
                 groups=1,
                 # 基础宽度
                 base_width=64,
                 # 卷积层类型
                 conv_layer=None,
                 # 归一化层类型
                 norm_layer=None,
                 # 激活层类型
                 act_layer=None
                 ):
        super(AttentionResNetBottleneck, self).__init__()
        assert with_attention in (0, 1)
        assert attention_type in ['GlobalContextBlock2D',
                                  'SimplifiedNonLocal2DEmbeddedGaussian',
                                  'NonLocal2DEmbeddedGaussian',
                                  'SqueezeAndExcitationBlock2D']

        if conv_layer is None:
            conv_layer = nn.Conv2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU

        self.downsample = downsample

        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv_layer(inplanes, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(width)

        self.conv2 = conv_layer(width, width, kernel_size=3, stride=stride, padding=1, bias=False, groups=groups)
        self.bn2 = norm_layer(width)

        self.conv3 = conv_layer(width, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = act_layer(inplace=True)

        self.with_attention = True if with_attention == 1 else 0
        self.attention_type = attention_type
        if self.with_attention:
            self.attention_block = self._make_attention_block(planes * self.expansion, reduction)
        else:
            self.attention_block = None

    def _make_attention_block(self, inplanes, reduction):
        if self.attention_type == 'GlobalContextBlock2D':
            return GlobalContextBlock2D(in_channels=inplanes, reduction=reduction)
        elif self.attention_type == 'SqueezeAndExcitationBlock2D':
            return SqueezeAndExcitationBlock2D(in_channels=inplanes, reduction=reduction)
        elif self.attention_type == 'NonLocal2DEmbeddedGaussian':
            return NonLocal2DEmbeddedGaussian(in_channels=inplanes)
        elif self.attention_type == 'SimplifiedNonLocal2DEmbeddedGaussian':
            return SimplifiedNonLocal2DEmbeddedGaussian(in_channels=inplanes)
        else:
            raise ValueError('no matching type')

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

        if self.with_attention and self.attention_type in ['GlobalContextBlock2D', 'SqueezeAndExcitationBlock2D']:
            out = self.attention_block(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        if self.with_attention and self.attention_type in ['NonLocal2DEmbeddedGaussian',
                                                           'SimplifiedNonLocal2DEmbeddedGaussian']:
            out = self.attention_block(out)

        return out

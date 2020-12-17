# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午2:38
@file: basicblock.py
@author: zj
@description: 
"""

import torch.nn as nn

from zcls.model.layers.global_context_block import GlobalContextBlock2D
from zcls.model.layers.squeeze_and_excitation_block import SqueezeAndExcitationBlock2D
from zcls.model.layers.non_local_embedded_gaussian import NonLocal2DEmbeddedGaussian
from zcls.model.layers.simplified_non_local_embedded_gaussian import SimplifiedNonLocal2DEmbeddedGaussian


class AttentionResNetBasicBlock(nn.Module):
    """
    使用两个3x3卷积，如果进行下采样，那么使用第一个卷积层对输入空间尺寸进行减半操作
    标准SE操作：在残差块操作后（after1x1）嵌入SE操作；
    """
    expansion = 1

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
                 # 卷积层类型
                 conv_layer=None,
                 # 归一化层类型
                 norm_layer=None,
                 # 激活层类型
                 act_layer=None
                 ):
        super(AttentionResNetBasicBlock, self).__init__()
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

        self.conv1 = conv_layer(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)

        self.conv2 = conv_layer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)

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

# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午2:38
@file: basicblock.py
@author: zj
@description: 
"""
from abc import ABC

import torch.nn as nn

from zcls.model.layers.place_holder import PlaceHolder
from zcls.model.layers.global_context_block import GlobalContextBlock2D
from zcls.model.layers.squeeze_and_excitation_block import SqueezeAndExcitationBlock2D
from zcls.model.layers.non_local_embedded_gaussian import NonLocal2DEmbeddedGaussian
from zcls.model.layers.simplified_non_local_embedded_gaussian import SimplifiedNonLocal2DEmbeddedGaussian


class AttentionResNetBasicBlock(nn.Module, ABC):
    """
    使用两个3x3卷积，如果进行下采样，那么使用第一个卷积层对输入空间尺寸进行减半操作
    对于Squeeze-And-Excitation或者Global Context操作，在残差连接中（after1x1）嵌入；
    对于NonLocal或者SimplifiedNonLoal，在Block完成计算后嵌入。
    """
    expansion = 1

    def __init__(self,
                 # 输入通道数
                 in_planes,
                 # 输出通道数
                 out_planes,
                 # 步长
                 stride=1,
                 # 下采样
                 down_sample=None,
                 # cardinality
                 groups=1,
                 # 基础宽度
                 base_width=64,
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
        if groups != 1 or base_width != 64:
            raise ValueError('AttentionResNetBasicBlock only supports groups=1 and base_width=64')

        if conv_layer is None:
            conv_layer = nn.Conv2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU

        self.down_sample = down_sample

        self.conv1 = conv_layer(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(out_planes)

        self.conv2 = conv_layer(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(out_planes)

        self.relu = act_layer(inplace=True)

        if with_attention == 0:
            self.attention1 = PlaceHolder()
            self.attention2 = PlaceHolder()
        else:
            if attention_type in ['SqueezeAndExcitationBlock2D', 'GlobalContextBlock2D']:
                self.attention1 = self._make_attention_block(out_planes * self.expansion, reduction, attention_type)
                self.attention2 = PlaceHolder()
            if attention_type in ['NonLocal2DEmbeddedGaussian', 'SimplifiedNonLocal2DEmbeddedGaussian']:
                self.attention1 = PlaceHolder()
                self.attention2 = self._make_attention_block(out_planes * self.expansion, reduction, attention_type)

    def _make_attention_block(self, in_planes, reduction, attention_type):
        if attention_type == 'GlobalContextBlock2D':
            return GlobalContextBlock2D(in_channels=in_planes, reduction=reduction)
        elif attention_type == 'SqueezeAndExcitationBlock2D':
            return SqueezeAndExcitationBlock2D(in_channels=in_planes, reduction=reduction)
        elif attention_type == 'NonLocal2DEmbeddedGaussian':
            return NonLocal2DEmbeddedGaussian(in_channels=in_planes)
        elif attention_type == 'SimplifiedNonLocal2DEmbeddedGaussian':
            return SimplifiedNonLocal2DEmbeddedGaussian(in_channels=in_planes)
        else:
            raise ValueError('no matching type')

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.attention1(out)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        out = self.attention2(out)

        return out
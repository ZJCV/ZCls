# -*- coding: utf-8 -*-

"""
@date: 2020/12/2 下午9:38
@file: mobilenetv1_backbone.py
@author: zj
@description: 
"""
from abc import ABC

import torch.nn as nn

from .mobilenetv1_block import MobileNetV1Block


class MobileNetV1Backbone(nn.Module, ABC):

    def __init__(self,
                 # 输入通道数
                 in_planes=3,
                 # 第一个卷积层通道数,
                 base_planes=32,
                 # 后续深度卷积层通道数
                 layer_planes=(64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024),
                 # 卷积步长
                 strides=(1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 2),
                 # 卷积层类型
                 conv_layer=None,
                 # 归一化层类型
                 norm_layer=None,
                 # 激活层类型
                 act_layer=None
                 ):
        super(MobileNetV1Backbone, self).__init__()
        assert len(strides) == len(layer_planes)

        if conv_layer is None:
            conv_layer = nn.Conv2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU

        self._make_stem(in_planes,
                        base_planes,
                        conv_layer,
                        norm_layer,
                        act_layer)
        self.layer_num = self._make_dw(base_planes,
                                       layer_planes,
                                       strides,
                                       conv_layer,
                                       norm_layer,
                                       act_layer)

        self.init_weights()

    def _make_stem(self,
                   in_planes,
                   base_planes,
                   conv_layer,
                   norm_layer,
                   act_layer
                   ):
        self.conv1 = conv_layer(in_planes, base_planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = norm_layer(base_planes)
        self.relu = act_layer(inplace=True)

    def _make_dw(self,
                 base_planes,
                 layer_planes,
                 strides,
                 conv_layer,
                 norm_layer,
                 act_layer
                 ):
        layer_num = len(layer_planes)

        in_planes = base_planes
        padding = 1
        for i, (layer_plane, stride) in enumerate(zip(layer_planes, strides)):
            if i == (layer_num - 1):
                # 最后一次深度卷积操作不进行空间尺寸下采样
                padding = 4
            dw_layer = MobileNetV1Block(int(in_planes),
                                        int(layer_plane),
                                        stride=stride,
                                        padding=padding,
                                        conv_layer=conv_layer,
                                        norm_layer=norm_layer,
                                        act_layer=act_layer
                                        )
            in_planes = layer_plane
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, dw_layer)

        return layer_num

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        for i in range(self.layer_num):
            layer = self.__getattr__(f'layer{i + 1}')
            x = layer(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

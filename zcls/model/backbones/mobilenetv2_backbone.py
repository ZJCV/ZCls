# -*- coding: utf-8 -*-

"""
@date: 2020/12/3 下午8:52
@file: mobilenetv2_backbone.py
@author: zj
@description: 
"""

import torch.nn as nn
from torchvision.models import mobilenet_v2

from .mobilenetv2_block import MobileNetV2Block


class MobileNetV2Backbone(nn.Module):

    def __init__(self,
                 # 输入通道数
                 inplanes=3,
                 # 输出通道数
                 planes=1280,
                 # 第一个卷积层通道数
                 base_channel=32,
                 # 第一个卷积层步长
                 stride=2,
                 # 第一个卷积层零填充
                 padding=1,
                 # 反向残差块设置
                 inverted_residual_setting=None,
                 # 卷积层类型
                 conv_layer=None,
                 # 归一化层类型
                 norm_layer=None,
                 # 激活层类型
                 act_layer=None,
                 ):
        super(MobileNetV2Backbone, self).__init__()

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
        else:
            assert len(inverted_residual_setting[0]) == 4

        if conv_layer is None:
            conv_layer = nn.Conv2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU6

        # 输入通道数
        self.inplanes = inplanes
        # 输出通道数
        self.planes = planes
        # 第一个卷积层通道数
        self.base_channel = base_channel
        # 第一个卷积层步长
        self.stride = stride
        # 第一个卷积层零填充
        self.padding = padding
        # 卷积层类型
        self.conv_layer = conv_layer
        # 归一化层类型
        self.norm_layer = norm_layer
        # 激活层类型
        self.act_layer = act_layer

        inplanes = self.inplanes
        planes = self.base_channel
        self.first_stem = self._make_stem(inplanes,
                                          planes,
                                          kernet_size=3,
                                          stride=stride,
                                          padding=padding,
                                          bias=False)

        inplanes = planes
        for i, (t, c, n, s) in enumerate(inverted_residual_setting):
            planes = c
            layer = MobileNetV2Block(inplanes,
                                     planes,
                                     t=t,
                                     n=n,
                                     stride=s,
                                     conv_layer=self.conv_layer,
                                     norm_layer=self.norm_layer,
                                     act_layer=self.act_layer)
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, layer)
            inplanes = planes
        planes = self.planes
        self.last_stem = self._make_stem(inplanes,
                                         planes,
                                         kernet_size=1,
                                         stride=1,
                                         padding=0,
                                         bias=False)

        self.layer_num = len(inverted_residual_setting)
        self._init_weights()

    def _make_stem(self,
                   inplanes,
                   planes,
                   kernet_size=1,
                   stride=1,
                   padding=0,
                   bias=False):
        return nn.Sequential(
            self.conv_layer(inplanes, planes, kernel_size=kernet_size,
                            stride=stride, padding=padding, bias=bias),
            self.norm_layer(planes),
            self.act_layer(inplace=True)
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.first_stem(x)

        for i in range(self.layer_num):
            layer = self.__getattr__(f'layer{i + 1}')
            x = layer(x)

        x = self.last_stem(x)
        return x

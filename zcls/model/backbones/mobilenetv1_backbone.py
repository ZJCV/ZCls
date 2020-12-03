# -*- coding: utf-8 -*-

"""
@date: 2020/12/2 下午9:38
@file: mobilenetv1_backbone.py
@author: zj
@description: 
"""

import torch.nn as nn

from .mobilenetv1_block import MobileNetV1Block


class MobileNetV1Backbone(nn.Module):

    def __init__(self,
                 # 输入通道数
                 inplanes=3,
                 # 第一个卷积层通道数,
                 base_channel=32,
                 # 后续深度卷积层通道数
                 channels=(64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024),
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
        assert len(strides) == len(channels)

        if conv_layer is None:
            conv_layer = nn.Conv2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU

        self.inplanes = inplanes
        self.base_channel = base_channel
        self.channels = channels
        self.strides = strides
        self.conv_layer = conv_layer
        self.norm_layer = norm_layer
        self.act_layer = act_layer

        self._make_stem()
        self._make_dw()

        self._init_weights()

    def _make_stem(self):
        self.conv1 = self.conv_layer(self.inplanes, self.base_channel, kernel_size=3,
                                     stride=2, padding=1, bias=False)
        self.bn1 = self.norm_layer(self.base_channel)
        self.relu = self.act_layer(inplace=True)

    def _make_dw(self):
        self.layer_num = len(self.channels)

        inplanes = self.base_channel
        padding = 1
        for i, (channel, stride) in enumerate(zip(self.channels, self.strides)):
            if i == (self.layer_num - 1):
                # 最后一次深度卷积操作不下采样空间尺寸
                padding = 4
            dw_layer = MobileNetV1Block(inplanes,
                                        channel,
                                        stride=stride,
                                        padding=padding,
                                        conv_layer=self.conv_layer,
                                        norm_layer=self.norm_layer,
                                        act_layer=self.act_layer
                                        )
            inplanes = channel
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, dw_layer)

    def _init_weights(self):
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

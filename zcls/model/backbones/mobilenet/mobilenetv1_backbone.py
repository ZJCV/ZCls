# -*- coding: utf-8 -*-

"""
@date: 2020/12/2 下午9:38
@file: mobilenetv1_backbone.py
@author: zj
@description: 
"""

from abc import ABC
import torch.nn as nn

from zcls.model import registry
from zcls.model.init_helper import init_weights
from zcls.model.conv_helper import get_conv
from zcls.model.norm_helper import get_norm
from zcls.model.act_helper import get_act
from ..misc import make_divisible
from .mobilenetv1_block import MobileNetV1Block


def make_stem(in_planes,
              base_planes,
              conv_layer,
              norm_layer,
              act_layer
              ):
    return nn.Sequential(
        conv_layer(in_planes, base_planes, kernel_size=3, stride=2, padding=1, bias=False),
        norm_layer(base_planes),
        act_layer(inplace=True)
    )


class MobileNetV1Backbone(nn.Module, ABC):

    def __init__(self,
                 in_channels=3,
                 base_channels=32,
                 layer_channels=(64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024),
                 strides=(1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 2),
                 conv_layer=None,
                 norm_layer=None,
                 act_layer=None
                 ):
        """
        :param in_channels: 输入通道数
        :param base_channels: 第一个卷积层通道数
        :param layer_channels: 后续深度卷积层通道数
        :param strides: 卷积步长
        :param conv_layer: 卷积层类型
        :param norm_layer: 归一化层类型
        :param act_layer: 激活层类型
        """
        super(MobileNetV1Backbone, self).__init__()
        assert len(strides) == len(layer_channels)

        if conv_layer is None:
            conv_layer = nn.Conv2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU

        self.stem = make_stem(in_channels,
                              base_channels,
                              conv_layer,
                              norm_layer,
                              act_layer)
        self.layer_num = self._make_dw(base_channels,
                                       layer_channels,
                                       strides,
                                       conv_layer,
                                       norm_layer,
                                       act_layer)
        init_weights(self.modules())

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

    def _forward_impl(self, x):
        x = self.stem(x)

        for i in range(self.layer_num):
            layer = self.__getattr__(f'layer{i + 1}')
            x = layer(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


@registry.Backbone.register('MobileNetV1')
def build_mbv1_backbone(cfg):
    in_planes = cfg.MODEL.BACKBONE.IN_PLANES
    base_planes = cfg.MODEL.BACKBONE.BASE_PLANES
    layer_planes = cfg.MODEL.BACKBONE.LAYER_PLANES
    strides = cfg.MODEL.BACKBONE.STRIDES
    conv_layer = get_conv(cfg)
    norm_layer = get_norm(cfg)
    act_layer = get_act(cfg)
    width_multiplier = cfg.MODEL.COMPRESSION.WIDTH_MULTIPLIER
    round_nearest = cfg.MODEL.COMPRESSION.ROUND_NEAREST

    base_planes = make_divisible(base_planes * width_multiplier, round_nearest)
    layer_planes = [make_divisible(layer_plane * width_multiplier, round_nearest) for layer_plane in layer_planes]

    return MobileNetV1Backbone(
        in_channels=in_planes,
        base_channels=base_planes,
        layer_channels=layer_planes,
        strides=strides,
        conv_layer=conv_layer,
        norm_layer=norm_layer,
        act_layer=act_layer
    )

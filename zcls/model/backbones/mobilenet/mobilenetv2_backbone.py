# -*- coding: utf-8 -*-

"""
@date: 2020/12/3 下午8:52
@file: mobilenetv2_backbone.py
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
from .mobilenetv2_block import MobileNetV2Block


def make_stem(in_planes,
              out_planes,
              kernel_size,
              stride,
              conv_layer,
              norm_layer,
              act_layer
              ):
    padding = 1 if stride == 2 else 0
    return nn.Sequential(
        conv_layer(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        norm_layer(out_planes),
        act_layer(inplace=True)
    )


class MobileNetV2Backbone(nn.Module, ABC):

    def __init__(self,
                 in_channels=3,
                 out_channels=1280,
                 base_channels=32,
                 stride=2,
                 padding=1,
                 inverted_residual_setting=None,
                 conv_layer=None,
                 norm_layer=None,
                 act_layer=None,
                 ):
        """
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param base_channels: 第一个卷积层通道数
        :param stride: 第一个卷积层步长
        :param padding: 第一个卷积层零填充
        :param inverted_residual_setting: 反向残差块设置
        :param conv_layer: 卷积层类型
        :param norm_layer: 归一化层类型
        :param act_layer: 激活层类型
        """
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

        kernel_size = 3
        stride = 2
        self.first_stem = make_stem(in_channels,
                                    base_channels,
                                    kernel_size,
                                    stride,
                                    conv_layer,
                                    norm_layer,
                                    act_layer
                                    )

        in_channels = base_channels
        for i, (t, c, n, s) in enumerate(inverted_residual_setting):
            feature_dims = c
            layer = MobileNetV2Block(in_channels,
                                     feature_dims,
                                     expansion_rate=t,
                                     repeat=n,
                                     stride=s,
                                     conv_layer=conv_layer,
                                     norm_layer=norm_layer,
                                     act_layer=act_layer)
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, layer)
            in_channels = feature_dims
        kernel_size = 1
        stride = 1
        self.last_stem = make_stem(in_channels,
                                   out_channels,
                                   kernel_size,
                                   stride,
                                   conv_layer,
                                   norm_layer,
                                   act_layer)

        self.layer_num = len(inverted_residual_setting)

        init_weights(self.modules())

    def forward(self, x):
        x = self.first_stem(x)

        for i in range(self.layer_num):
            layer = self.__getattr__(f'layer{i + 1}')
            x = layer(x)

        x = self.last_stem(x)
        return x


@registry.Backbone.register('MobileNetV2')
def build_mbv2_backbone(cfg):
    in_channels = cfg.MODEL.BACKBONE.IN_PLANES
    base_channels = cfg.MODEL.BACKBONE.BASE_PLANES
    out_channels = cfg.MODEL.HEAD.FEATURE_DIMS
    round_nearest = cfg.MODEL.COMPRESSION.ROUND_NEAREST
    width_multiplier = cfg.MODEL.COMPRESSION.WIDTH_MULTIPLIER
    conv_layer = get_conv(cfg)
    norm_layer = get_norm(cfg)
    act_layer = get_act(cfg)

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

    base_channels = make_divisible(base_channels * width_multiplier, round_nearest)
    for i in range(len(inverted_residual_setting)):
        channel = inverted_residual_setting[i][1]
        inverted_residual_setting[i][1] = make_divisible(channel * width_multiplier, round_nearest)
    # out_channels = make_divisible(out_channels * width_multiplier, round_nearest)

    return MobileNetV2Backbone(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        inverted_residual_setting=inverted_residual_setting,
        conv_layer=conv_layer,
        norm_layer=norm_layer,
        act_layer=act_layer
    )

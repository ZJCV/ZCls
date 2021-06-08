# -*- coding: utf-8 -*-

"""
@date: 2020/12/30 下午4:28
@file: mnasnet_unit.py
@author: zj
@description:
"""
from abc import ABC

import copy
import torch.nn as nn

from zcls.model import registry
from zcls.model.conv_helper import get_conv
from zcls.model.norm_helper import get_norm
from zcls.model.act_helper import get_act
from ..misc import round_to_multiple_of
from .mobilenetv3_unit import MobileNetV3Unit

arch_settings = {
    'mobilenetv3-large': [16, 960, 1280,
                          [
                              # kernel_size, stride, inner_planes, with_attention, non-linearity, out_planes
                              [3, 1, 16, 0, 'RE', 16],
                              [3, 2, 64, 0, 'RE', 24],
                              [3, 1, 72, 0, 'RE', 24],
                              [5, 2, 72, 1, 'RE', 40],
                              [5, 1, 120, 1, 'RE', 40],
                              [5, 1, 120, 1, 'RE', 40],
                              [3, 2, 240, 0, 'HS', 80],
                              [3, 1, 200, 0, 'HS', 80],
                              [3, 1, 184, 0, 'HS', 80],
                              [3, 1, 184, 0, 'HS', 80],
                              [3, 1, 480, 1, 'HS', 112],
                              [3, 1, 672, 1, 'HS', 112],
                              [5, 2, 672, 1, 'HS', 160],
                              [5, 1, 960, 1, 'HS', 160],
                              [5, 1, 960, 1, 'HS', 160],
                          ]],
    'mobilenetv3-small': [16, 576, 1024,
                          [
                              # kernel_size, stride, inner_planes, with_attention, non-linearity, out_planes
                              [3, 2, 16, 1, 'RE', 16],
                              [3, 2, 72, 0, 'RE', 24],
                              [3, 1, 88, 0, 'RE', 24],
                              [5, 2, 96, 1, 'HS', 40],
                              [5, 1, 240, 1, 'HS', 40],
                              [5, 1, 240, 1, 'HS', 40],
                              [5, 1, 120, 1, 'HS', 48],
                              [5, 1, 144, 1, 'HS', 48],
                              [5, 2, 288, 1, 'HS', 96],
                              [5, 1, 576, 1, 'HS', 96],
                              [5, 1, 576, 1, 'HS', 96]
                          ]]
}


def relu_or_hswish(name):
    if name == 'RE':
        return nn.ReLU
    elif name == 'HS':
        return nn.Hardswish
    else:
        raise IOError(f'{name} does not exist')


def make_stem(in_planes,
              base_planes,
              inner_planes,
              out_planes,
              conv_layer,
              norm_layer,
              act_layer):
    first_stem = nn.Sequential(
        conv_layer(in_planes, base_planes, kernel_size=3, stride=2, padding=1, bias=False),
        norm_layer(base_planes),
        act_layer(inplace=True)
    )

    last_stem = nn.Sequential(
        conv_layer(inner_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
        norm_layer(out_planes),
        act_layer(inplace=True)
    )

    return first_stem, last_stem


class MobileNetV3Backbone(nn.Module, ABC):

    def __init__(self,
                 in_channels=3,
                 base_channels=16,
                 out_channels=960,
                 layer_setting=None,
                 width_multiplier=1.,
                 round_nearest=8,
                 with_attention=True,
                 reduction=4,
                 attention_type='SqueezeAndExcitationBlock2D',
                 conv_layer=None,
                 norm_layer=None,
                 act_layer=None,
                 sigmoid_type=None
                 ):
        """
        :param in_channels: 输入通道数
        :param base_channels: 基础通道数
        :param out_channels: 输出通道数
        :param width_multiplier: 宽度乘法器
        :param round_nearest: 设置每一层通道数均为8的倍数
        :param with_attention: 是否使用注意力模块
        :param reduction: 衰减率
        :param attention_type: 注意力模块类型
        :param conv_layer: 卷积层类型
        :param norm_layer: 归一化层类型
        :param act_layer: 激活层类型
        :param sigmoid_type: sigmoid类型
        """
        super(MobileNetV3Backbone, self).__init__()

        if conv_layer is None:
            conv_layer = nn.Conv2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.Hardswish
        if sigmoid_type is None:
            sigmoid_type = 'HSigmoid'
        block_layer = MobileNetV3Unit

        if layer_setting is None:
            layer_setting = [
                # kernel_size, stride, inner_planes, with_attention_2, non-linearity, out_planes
                [3, 1, 16, 0, 'RE', 16],
                [3, 2, 64, 0, 'RE', 24],
                [3, 1, 72, 0, 'RE', 24],
                [5, 2, 72, 1, 'RE', 40],
                [5, 1, 120, 1, 'RE', 40],
                [5, 1, 120, 1, 'RE', 40],
                [3, 2, 240, 0, 'HS', 80],
                [3, 1, 200, 0, 'HS', 80],
                [3, 1, 184, 0, 'HS', 80],
                [3, 1, 184, 0, 'HS', 80],
                [3, 1, 480, 1, 'HS', 112],
                [3, 1, 672, 1, 'HS', 112],
                [5, 2, 672, 1, 'HS', 160],
                [5, 1, 960, 1, 'HS', 160],
                [5, 1, 960, 1, 'HS', 160],
            ]

        base_channels = round_to_multiple_of(base_channels * width_multiplier, round_nearest)
        out_channels = round_to_multiple_of(out_channels * width_multiplier, round_nearest)

        for i in range(len(layer_setting)):
            # 缩放膨胀通道数
            layer_setting[i][2] = round_to_multiple_of(layer_setting[i][2], round_nearest)
            # 缩放输出通道数
            layer_setting[i][-1] = round_to_multiple_of(layer_setting[i][-1] * width_multiplier, round_nearest)

        self.first_stem, self.last_stem = make_stem(in_channels,
                                                    base_channels,
                                                    layer_setting[-1][-1],
                                                    out_channels,
                                                    conv_layer,
                                                    norm_layer,
                                                    act_layer)

        features = list()
        in_channels = base_channels
        for i, (kernel_size, stride, inner_planes, with_attention_2, non_linearity, out_channels) in enumerate(
                layer_setting):
            act_layer = relu_or_hswish(non_linearity)
            features.append(block_layer(in_channels,
                                        inner_planes,
                                        out_channels,
                                        stride=stride,
                                        kernel_size=kernel_size,
                                        with_attention=with_attention_2 and with_attention,
                                        reduction=reduction,
                                        attention_type=attention_type,
                                        conv_layer=conv_layer,
                                        norm_layer=norm_layer,
                                        act_layer=act_layer,
                                        sigmoid_type=sigmoid_type
                                        ))
            in_channels = out_channels
        self.add_module('features', nn.Sequential(*features))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="sigmoid")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.first_stem(x)
        x = self.features(x)
        x = self.last_stem(x)
        return x


@registry.Backbone.register('MobileNetV3')
def build_mbv3_backbone(cfg):
    arch = cfg.MODEL.BACKBONE.ARCH
    in_channels = cfg.MODEL.BACKBONE.IN_PLANES
    norm_layer = get_norm(cfg)
    # compression
    width_multiplier = cfg.MODEL.COMPRESSION.WIDTH_MULTIPLIER
    round_nearest = cfg.MODEL.COMPRESSION.ROUND_NEAREST
    # attention
    with_attention = cfg.MODEL.ATTENTION.WITH_ATTENTION
    reduction = cfg.MODEL.ATTENTION.REDUCTION
    attention_type = cfg.MODEL.ATTENTION.ATTENTION_TYPE
    # conv
    conv_layer = get_conv(cfg)
    # act
    act_layer = get_act(cfg)
    sigmoid_type = cfg.MODEL.ACT.SIGMOID_TYPE

    base_channels, feature_dims, inner_dims, layer_setting = copy.deepcopy(arch_settings[arch])

    return MobileNetV3Backbone(
        in_channels=in_channels,
        base_channels=base_channels,
        out_channels=feature_dims,
        layer_setting=layer_setting,
        width_multiplier=width_multiplier,
        round_nearest=round_nearest,
        with_attention=with_attention,
        reduction=reduction,
        attention_type=attention_type,
        conv_layer=conv_layer,
        norm_layer=norm_layer,
        act_layer=act_layer,
        sigmoid_type=sigmoid_type
    )

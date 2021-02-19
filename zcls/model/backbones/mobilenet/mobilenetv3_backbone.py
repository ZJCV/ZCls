# -*- coding: utf-8 -*-

"""
@date: 2020/12/30 下午4:28
@file: mnasnet_unit.py
@author: zj
@description:
"""
from abc import ABC

import torch.nn as nn

from zcls.model.layers.hard_swish_wrapper import HardswishWrapper
from .mobilenetv3_unit import MobileNetV3Uint, BN_MOMENTUM


def _round_to_multiple_of(val, divisor, round_up_bias=0.9):
    """ Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88. """
    assert 0.0 < round_up_bias < 1.0
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor


def relu_or_hswish(name):
    if name == 'RE':
        return nn.ReLU
    elif name == 'HS':
        return HardswishWrapper
    else:
        raise IOError(f'{name} does not exist')


class MobileNetV3Backbone(nn.Module, ABC):

    def __init__(self,
                 # 输入通道数
                 in_planes=3,
                 # 基础通道数
                 base_planes=16,
                 # 输出通道数
                 out_planes=960,
                 # 宽度乘法器
                 width_multiplier=1.,
                 # 设置每一层通道数均为8的倍数
                 round_nearest=8,
                 # 是否使用注意力模块
                 with_attention=True,
                 # 衰减率
                 reduction=4,
                 # 注意力模块类型
                 attention_type='SqueezeAndExcitationBlock2D',
                 # 卷积层类型
                 conv_layer=None,
                 # 归一化层类型
                 norm_layer=None,
                 # 激活层类型
                 act_layer=None,
                 # sigmoid类型
                 sigmoid_type=None
                 ):
        super(MobileNetV3Backbone, self).__init__()

        if conv_layer is None:
            conv_layer = nn.Conv2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = HardswishWrapper
        if sigmoid_type is None:
            sigmoid_type = 'HSigmoid'
        block_layer = MobileNetV3Uint

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

        base_planes = _round_to_multiple_of(base_planes * width_multiplier, round_nearest)
        # 参考Torchvision MnasNet实现，不对输出特征维度进行缩放
        # out_planes = _round_to_multiple_of(out_planes * width_multiplier, round_nearest)
        for i in range(len(layer_setting)):
            # 缩放膨胀通道数
            layer_setting[i][2] = _round_to_multiple_of(layer_setting[i][2] * width_multiplier, round_nearest)
            # 缩放输出通道数
            layer_setting[i][-1] = _round_to_multiple_of(layer_setting[i][-1] * width_multiplier, round_nearest)

        self.make_stem(in_planes,
                       base_planes,
                       layer_setting[-1][-1],
                       out_planes,
                       conv_layer,
                       norm_layer,
                       act_layer)

        features = list()
        in_planes = base_planes
        for i, (kernel_size, stride, inner_planes, with_attention_2, non_linearity, out_planes) in enumerate(
                layer_setting):
            act_layer = relu_or_hswish(non_linearity)
            features.append(block_layer(in_planes,
                                        inner_planes,
                                        out_planes,
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
            in_planes = out_planes
        self.add_module('features', nn.Sequential(*features))

        self._init_weights()

    def make_stem(self,
                  in_planes,
                  base_planes,
                  inner_planes,
                  out_planes,
                  conv_layer,
                  norm_layer,
                  act_layer):
        self.first_stem = nn.Sequential(
            conv_layer(in_planes, base_planes, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(base_planes, momentum=BN_MOMENTUM),
            act_layer(inplace=True)
        )

        self.last_stem = nn.Sequential(
            conv_layer(inner_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
            norm_layer(out_planes, momentum=BN_MOMENTUM),
            act_layer(inplace=True)
        )

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

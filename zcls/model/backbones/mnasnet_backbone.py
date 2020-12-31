# -*- coding: utf-8 -*-

"""
@date: 2020/12/30 下午4:28
@file: mnasnet_unit.py
@author: zj
@description:
"""
from abc import ABC

import torch.nn as nn

from .mnasnet_unit import MNASNetUint, BN_MOMENTUM


def _round_to_multiple_of(val, divisor, round_up_bias=0.9):
    """ Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88. """
    assert 0.0 < round_up_bias < 1.0
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor


class MNASNetBackbone(nn.Module, ABC):
    """
    MnasNet-A1实现
    参考Torchvision mnasnet实现，在Backbone之后添加一个Conv1x1-BN-ReLU，最终输出特征维度为1280
    """

    def __init__(self,
                 # 输入通道数
                 in_planes=3,
                 # 输出通道数
                 out_planes=1280,
                 # 参数设置
                 stage_setting=None,
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
                 act_layer=None
                 ):
        super(MNASNetBackbone, self).__init__()

        if conv_layer is None:
            conv_layer = nn.Conv2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU
        block_layer = MNASNetUint

        base_planes = 32
        if stage_setting is None:
            stage_setting = [
                # kernel_size, stride, expansion_rate, with_attention_2, repeated, out_planes
                [3, 1, 1, 0, 1, 16],
                [3, 2, 6, 0, 2, 24],
                [5, 2, 3, 1, 3, 40],
                [3, 2, 6, 0, 4, 80],
                [3, 1, 6, 1, 2, 112],
                [5, 2, 6, 1, 3, 160],
                [3, 1, 6, 0, 1, 320],
            ]

        base_planes = _round_to_multiple_of(base_planes * width_multiplier, round_nearest)
        # 参考Torchvision实现，不对输出特征维度进行缩放
        # out_planes = _round_to_multiple_of(out_planes * width_multiplier, round_nearest)
        for i in range(len(stage_setting)):
            stage_setting[i][-1] = _round_to_multiple_of(stage_setting[i][-1] * width_multiplier, round_nearest)

        self.make_stem(in_planes,
                       base_planes,
                       stage_setting[-1][-1],
                       out_planes,
                       conv_layer,
                       norm_layer,
                       act_layer)

        in_planes = base_planes
        for i, (kernel_size, stride, expansion_rate, with_attention_2, repeated, out_planes) in enumerate(
                stage_setting):
            features = list()
            for j in range(repeated):
                stride = stride if j == 0 else 1
                features.append(block_layer(in_planes,
                                            out_planes,
                                            stride=stride,
                                            kernel_size=kernel_size,
                                            expansion_rate=expansion_rate,
                                            with_attention=with_attention and with_attention_2,
                                            reduction=reduction,
                                            attention_type=attention_type,
                                            conv_layer=conv_layer,
                                            norm_layer=norm_layer,
                                            act_layer=act_layer
                                            ))
                in_planes = out_planes
            stage_name = f'stage{i + 1}'
            self.add_module(stage_name, nn.Sequential(*features))

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

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)

        x = self.last_stem(x)
        return x

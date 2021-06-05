# -*- coding: utf-8 -*-

"""
@date: 2021/6/3 下午8:31
@file: ghost_backbone.py
@author: zj
@description: 
"""

import copy
import torch.nn as nn

from zcls.model import registry
from zcls.model.conv_helper import get_conv
from zcls.model.norm_helper import get_norm
from zcls.model.act_helper import get_act
from .ghost_bottleneck import GhostBottleneck
from ..misc import round_to_multiple_of

arch_setting = [
    # kernel_size, expansion channels, output channels, squeeze-and-excitation reduction, stride
    # stage1
    [
        [3, 16, 16, 0, 1]
    ],
    # stage2
    [
        [3, 48, 24, 0, 2]
    ],
    # stage3
    [
        [3, 72, 24, 0, 1]
    ],
    # stage4
    [
        [5, 72, 40, 4, 2]
    ],
    # stage5
    [
        [5, 120, 40, 4, 1]
    ],
    # stage6
    [
        [3, 240, 80, 0, 2]
    ],
    # stage7
    [
        [3, 200, 80, 0, 1],
        [3, 184, 80, 0, 1],
        [3, 184, 80, 0, 1],
        [3, 480, 112, 4, 1],
        [3, 672, 112, 4, 1]
    ],
    # stage8
    [
        [5, 672, 160, 4, 2]
    ],
    # stage9
    [
        [5, 960, 160, 0, 1],
        [5, 960, 160, 4, 1],
        [5, 960, 160, 0, 1],
        [5, 960, 160, 4, 1]
    ]
]


def make_stem(in_channels,
              out_channels,
              last_in_channels,
              last_out_channels,
              conv_layer,
              norm_layer,
              act_layer
              ):
    """
    :param in_channels: first stem输入通道数
    :param out_channels: first stem输出通道数
    :param last_in_channels: last stem输入通道数
    :param last_out_channels: last stem输出通道数
    :param conv_layer: 卷积层
    :param norm_layer: 池化层
    :param act_layer: 激活层
    """
    return nn.Sequential(
        conv_layer(in_channels, out_channels, kernel_size=(3, 3), stride=2, padding=1, bias=False),
        norm_layer(out_channels),
        act_layer(inplace=True),
    ), nn.Sequential(
        conv_layer(last_in_channels, last_out_channels, kernel_size=(1, 1), stride=1, padding=0, bias=False),
        norm_layer(last_out_channels),
        act_layer(inplace=True),
    )


class GhostBackbone(nn.Module):

    def __init__(self,
                 in_channels=3,
                 base_channels=16,
                 width_multiplier=1.0,
                 round_nearest=4,
                 attention_type='SqueezeAndExcitationBlock2D',
                 arch_configs=None,
                 block_layer=None,
                 conv_layer=None,
                 norm_layer=None,
                 act_layer=None,
                 **kwargs
                 ):
        """
        :param in_channels: 输入通道数
        :param base_channels: 基础通道数
        :param width_multiplier: 宽度乘法器
        :param round_nearest: 设置每一层通道数均为4的倍数
        :param attention_type: 注意力模块类型
        :param block_layer: 块类型
        :param conv_layer: 卷积层类型
        :param norm_layer: 归一化层类型
        :param act_layer: 激活层类型
        :param zero_init_residual: 零初始化残差连接
        :param kwargs: 其他参数
        """
        super(GhostBackbone, self).__init__()

        if arch_configs is None:
            arch_configs = copy.deepcopy(arch_setting)
        if block_layer is None:
            block_layer = GhostBottleneck
        if conv_layer is None:
            conv_layer = nn.Conv2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU

        base_channels = round_to_multiple_of(base_channels * width_multiplier, round_nearest)
        last_in_channels = round_to_multiple_of(arch_configs[-1][-1][2] * width_multiplier, round_nearest)
        last_out_channels = round_to_multiple_of(arch_configs[-1][-1][1] * width_multiplier, round_nearest)
        self.first_stem, self.last_stem = make_stem(in_channels,
                                              base_channels,
                                              last_in_channels,
                                              last_out_channels,
                                              conv_layer,
                                              norm_layer,
                                              act_layer)

        input_channels = base_channels
        # building inverted residual blocks
        stages = []
        for cfg in arch_configs:
            layers = []
            for kernel_size, expansion_size, out_channels, attention_reduction_rate, stride in cfg:
                hidden_channels = round_to_multiple_of(expansion_size * width_multiplier, round_nearest)
                output_channels = round_to_multiple_of(out_channels * width_multiplier, round_nearest)
                layers.append(block_layer(input_channels,
                                          hidden_channels,
                                          output_channels,
                                          stride=stride,
                                          kernel_size=kernel_size,
                                          with_attention=attention_reduction_rate > 1,
                                          reduction=attention_reduction_rate,
                                          attention_type=attention_type,
                                          )
                              )
                input_channels = output_channels
            stages.append(nn.Sequential(*layers))

        self.blocks = nn.Sequential(*stages)

    def forward(self, x):
        x = self.first_stem(x)
        x = self.blocks(x)
        x = self.last_stem(x)

        return x


@registry.Backbone.register('GhostNet')
def build_ghostnet_backbone(cfg):
    in_channels = cfg.MODEL.BACKBONE.IN_PLANES
    base_channels = cfg.MODEL.BACKBONE.BASE_PLANES
    # compression
    width_multiplier = cfg.MODEL.COMPRESSION.WIDTH_MULTIPLIER
    round_nearest = cfg.MODEL.COMPRESSION.ROUND_NEAREST
    # attention
    attention_type = cfg.MODEL.ATTENTION.ATTENTION_TYPE
    # conv
    conv_layer = get_conv(cfg)
    # norm
    norm_layer = get_norm(cfg)
    # act
    act_layer = get_act(cfg)

    return GhostBackbone(
        in_channels=in_channels,
        base_channels=base_channels,
        width_multiplier=width_multiplier,
        round_nearest=round_nearest,
        attention_type=attention_type,
        conv_layer=conv_layer,
        norm_layer=norm_layer,
        act_layer=act_layer,
    )

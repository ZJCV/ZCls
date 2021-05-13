# -*- coding: utf-8 -*-

"""
@date: 2021/2/2 下午4:33
@file: repvgg_backbone.py
@author: zj
@description: 
"""

import torch.nn as nn

from zcls.model import registry
from zcls.model.conv_helper import get_conv
from zcls.model.act_helper import get_act
from zcls.model.init_helper import init_weights
from zcls.model.attention_helper import make_attention_block

optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}

arch_settings = {
    # name: (num_blocks, width_multiplier, groups)
    'repvgg_a0': ((2, 4, 14, 1), (0.75, 2.5), {}),
    'repvgg_a1': ((2, 4, 14, 1), (1, 2.5), {}),
    'repvgg_a2': ((2, 4, 14, 1), (1.5, 2.75), {}),
    'repvgg_b0': ((4, 6, 16, 1), (1, 2.5), {}),
    'repvgg_b1': ((4, 6, 16, 1), (2, 4), {}),
    'repvgg_b1g2': ((4, 6, 16, 1), (2, 4), g2_map),
    'repvgg_b1g4': ((4, 6, 16, 1), (2, 4), g4_map),
    'repvgg_b2': ((4, 6, 16, 1), (2.5, 5), {}),
    'repvgg_b2g2': ((4, 6, 16, 1), (2.5, 5), g2_map),
    'repvgg_b2g4': ((4, 6, 16, 1), (2.5, 5), g4_map),
    'repvgg_b3': ((4, 6, 16, 1), (3, 5), {}),
    'repvgg_b3g2': ((4, 6, 16, 1), (3, 5), g2_map),
    'repvgg_b3g4': ((4, 6, 16, 1), (3, 5), g4_map),
    'repvgg_d2se': ((8, 14, 24, 1), (2.5, 5), {})
}


def make_stem(in_channels,
              base_channels,
              groups,
              with_attention,
              reduction,
              attention_type,
              attention_bias,
              conv_layer,
              act_layer
              ):
    """

    :param in_channels: 输入通道数
    :param base_channels: 卷积层输出通道数
    :param groups: 分组数
    :param conv_layer: 卷积层
    :param act_layer: 激活层类型
    :return:
    """
    return nn.Sequential(
        conv_layer(in_channels=in_channels, out_channels=base_channels, kernel_size=3, stride=2, padding=1,
                   groups=groups, bias=True),
        make_attention_block(base_channels, reduction,
                             attention_type, bias=attention_bias) if with_attention else nn.Identity(),
        act_layer(inplace=True)
    )


class RepVGGBackbone(nn.Module):

    def __init__(self,
                 in_channels=3,
                 base_channels=64,
                 layer_channels=(64, 128, 256, 512),
                 layer_blocks=(2, 4, 14, 1),
                 width_multipliers=(1., 1.),
                 down_samples=(1, 1, 1, 1),
                 groups=dict(),
                 with_attention=False,
                 reduction=16,
                 attention_type='SqueezeAndExcitationBlock2D',
                 attention_bias=False,
                 conv_layer=None,
                 act_layer=None,
                 ):
        """
        :param in_channels: 输入通道数
        :param base_channels: 基础通道数
        :param layer_channels: 每一层通道数
        :param layer_blocks: 每一层块个数
        :param width_multipliers: 宽度乘法器
        :param down_samples: 是否执行空间下采样
        :param groups: cardinality
        :param conv_layer: 卷积层类型
        :param act_layer: 激活层类型
        """
        super(RepVGGBackbone, self).__init__()
        assert len(layer_channels) == len(layer_blocks) == len(down_samples)
        assert len(width_multipliers) == 2
        assert isinstance(groups, dict)

        if conv_layer is None:
            conv_layer = nn.Conv2d
        if act_layer is None:
            act_layer = nn.ReLU

        self.with_attention = with_attention
        self.reduction = reduction
        self.attention_type = attention_type
        self.attention_bias = attention_bias

        self.cur_layer_idx = 0
        self.override_groups_map = groups
        cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)

        width_multiplier_a, width_multiplier_b = width_multipliers
        base_channels = min(int(base_channels), int(base_channels * width_multiplier_a))
        self.stage0 = make_stem(in_channels, base_channels, cur_groups,
                                with_attention, reduction, attention_type, attention_bias,
                                conv_layer, act_layer)

        in_channels = base_channels
        for i in range(len(layer_blocks)):
            width_multiplier = width_multiplier_a if i != (len(layer_blocks) - 1) else width_multiplier_b
            out_channels = int(layer_channels[i] * width_multiplier)
            stage = self._make_stage(in_channels,
                                     out_channels,
                                     layer_blocks[i],
                                     down_samples[i],
                                     conv_layer,
                                     act_layer
                                     )
            in_channels = out_channels
            stage_name = f'stage{i + 1}'
            self.add_module(stage_name, stage)

        init_weights(self.modules())

    def _make_stage(self,
                    in_channels,
                    out_channels,
                    block_num,
                    with_sample,
                    conv_layer,
                    act_layer,
                    ):
        """
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param block_num: 块个数
        :param with_sample: 是否执行空间下采样
        :param conv_layer: 卷积层类型
        :param act_layer: 激活层类型
        :return:
        """
        stride = 2 if with_sample else 1
        padding = 1 if stride == 2 else 0
        self.cur_layer_idx += 1
        cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)

        blocks = list()
        blocks.append(
            nn.Sequential(
                conv_layer(in_channels, out_channels, kernel_size=3,
                           stride=stride, padding=padding, groups=cur_groups, bias=True),
                make_attention_block(out_channels, self.reduction,
                                     self.attention_type, bias=self.attention_bias)
                if self.with_attention else nn.Identity(),
                act_layer(inplace=True)
            )
        )

        in_channels = out_channels
        stride = 1
        padding = 1

        for i in range(1, block_num):
            self.cur_layer_idx += 1
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(
                nn.Sequential(
                    conv_layer(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=padding, groups=cur_groups, bias=True),
                    make_attention_block(out_channels, self.reduction,
                                         self.attention_type, bias=self.attention_bias)
                    if self.with_attention else nn.Identity(),
                    act_layer(inplace=True)
                )
            )
        return nn.Sequential(*blocks)

    def _forward_impl(self, x):
        x = self.stage0(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


@registry.Backbone.register('RepVGG')
def build_repvgg_backbone(cfg):
    arch = cfg.MODEL.BACKBONE.ARCH
    in_channels = cfg.MODEL.BACKBONE.IN_PLANES
    base_channels = cfg.MODEL.BACKBONE.BASE_PLANES
    layer_planes = cfg.MODEL.BACKBONE.LAYER_PLANES
    down_samples = cfg.MODEL.BACKBONE.DOWNSAMPLES
    with_attention = cfg.MODEL.ATTENTION.WITH_ATTENTION
    reduction = cfg.MODEL.ATTENTION.REDUCTION
    attention_type = cfg.MODEL.ATTENTION.ATTENTION_TYPE
    attention_bias = cfg.MODEL.ATTENTION.BIAS
    conv_layer = get_conv(cfg)
    act_layer = get_act(cfg)

    num_blocks, width_multipliers, groups = arch_settings[arch]
    return RepVGGBackbone(
        in_channels=in_channels,
        base_channels=base_channels,
        layer_channels=layer_planes,
        layer_blocks=num_blocks,
        down_samples=down_samples,
        width_multipliers=width_multipliers,
        groups=groups,
        with_attention=with_attention,
        reduction=reduction,
        attention_type=attention_type,
        attention_bias=attention_bias,
        conv_layer=conv_layer,
        act_layer=act_layer,
    )

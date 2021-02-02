# -*- coding: utf-8 -*-

"""
@date: 2021/2/2 下午4:33
@file: repvgg_backbone.py
@author: zj
@description: 
"""

import torch.nn as nn

from ..init_helper import init_weights


class RepVGGBackbone(nn.Module):

    def __init__(self,
                 # 输入通道数
                 in_channels=3,
                 # 基础通道数,
                 base_channels=64,
                 # 每一层通道数
                 layer_planes=(64, 128, 256, 512),
                 # 每一层块个数
                 layer_blocks=(2, 4, 14, 1),
                 # 宽度乘法器
                 width_multipliers=(1., 1.),
                 # 是否执行空间下采样
                 down_samples=(1, 1, 1, 1),
                 # cardinality
                 groups=dict,
                 # 卷积层类型
                 conv_layer=None,
                 # 激活层类型
                 act_layer=None,
                 ):
        super(RepVGGBackbone, self).__init__()
        assert len(layer_planes) == len(layer_blocks) == len(down_samples)
        assert len(width_multipliers) == 2
        assert isinstance(groups, dict)

        if conv_layer is None:
            conv_layer = nn.Conv2d
        if act_layer is None:
            act_layer = nn.ReLU

        width_multiplier_a, width_multiplier_b = width_multipliers
        base_channels = min(int(base_channels), int(base_channels * width_multiplier_a))
        self._make_stem(in_channels, base_channels, conv_layer)

        self.cur_layer_idx = 1
        self.override_groups_map = groups
        in_planes = base_channels
        for i in range(len(layer_blocks)):
            width_multiplier = width_multiplier_a if i != (len(layer_blocks) - 1) else width_multiplier_b
            out_channels = int(layer_planes[i] * width_multiplier)
            stage = self._make_stage(in_planes,
                                     out_channels,
                                     layer_blocks[i],
                                     down_samples[i],
                                     conv_layer,
                                     act_layer
                                     )
            in_planes = out_channels
            stage_name = f'stage{i + 1}'
            self.add_module(stage_name, stage)

        self.init_weights()

    def _make_stem(self,
                   # 输入通道数
                   in_planes,
                   # 卷积层输出通道数
                   base_planes,
                   # 卷积层
                   conv_layer
                   ):
        self.stage0 = conv_layer(in_channels=in_planes, out_channels=base_planes, kernel_size=3, stride=2, padding=1)

    def _make_stage(self,
                    # 输入通道数
                    in_planes,
                    # 输出通道数
                    out_planes,
                    # 块个数
                    block_num,
                    # 是否执行空间下采样
                    with_sample,
                    # 卷积层类型
                    conv_layer,
                    # 激活层类型
                    act_layer,
                    ):
        stride = 2 if with_sample else 1
        padding = 1 if stride == 2 else 0
        cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)

        blocks = list()
        blocks.append(
            nn.Sequential(
                conv_layer(in_planes, out_planes, kernel_size=3,
                           stride=stride, padding=padding, groups=cur_groups, bias=True),
                act_layer(inplace=True)
            )
        )

        in_planes = out_planes
        stride = 1
        padding = 1

        for i in range(1, block_num):
            self.cur_layer_idx += 1
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(
                nn.Sequential(
                    conv_layer(in_planes, out_planes, kernel_size=3,
                               stride=stride, padding=padding, groups=cur_groups, bias=True),
                    act_layer(inplace=True)
                )
            )
        return nn.Sequential(*blocks)

    def init_weights(self):
        init_weights(self.modules())

    def _forward_impl(self, x):
        x = self.stage0(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

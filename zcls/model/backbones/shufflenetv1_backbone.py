# -*- coding: utf-8 -*-

"""
@date: 2020/12/3 下午8:52
@file: mobilenetv2_backbone.py
@author: zj
@description: 
"""
from abc import ABC

import torch.nn as nn
from .shufflenetv1_unit import ShuffleNetV1Unit


def make_stage(  # 输入通道数
        in_planes,
        # 输出通道数
        out_planes,
        # 分组数
        groups,
        # 块个数
        block_num,
        # 是否执行空间下采样
        with_down_sample,
        # 是否对第一个1x1逐点卷积执行分组操作
        with_groups,
        # 块类型
        block_layer,
        # 卷积层类型
        conv_layer,
        # 归一化层类型
        norm_layer,
        # 激活层类型
        act_layer,
):
    with_groups = with_groups if isinstance(with_groups, tuple) else [with_groups] * block_num
    assert len(with_groups) == block_num
    stride = 2 if with_down_sample else 1
    down_sample = nn.AvgPool2d(3, stride=stride, padding=1) if stride == 2 else None

    blocks = list()
    blocks.append(block_layer(
        in_planes, out_planes, groups, stride, down_sample, with_groups[0], conv_layer, norm_layer, act_layer))
    in_planes = out_planes

    groups = 1
    stride = 1
    down_sample = None
    for i in range(1, block_num):
        blocks.append(block_layer(
            in_planes, out_planes, groups, stride, down_sample, with_groups[i], conv_layer, norm_layer, act_layer))
    return nn.Sequential(*blocks)


def make_stem(in_planes,
              base_planes,
              conv_layer,
              norm_layer,
              act_layer
              ):
    return nn.Sequential(
        conv_layer(in_planes, base_planes, kernel_size=3, stride=2, padding=1, bias=False),
        norm_layer(base_planes),
        act_layer(inplace=True),
        nn.MaxPool2d(3, stride=2, padding=1)
    )


def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class ShuffleNetV1Backbone(nn.Module, ABC):

    def __init__(self,
                 # 输入通道数
                 in_planes=3,
                 # 第一个卷积层通道数
                 base_planes=24,
                 # 分组数
                 groups=8,
                 # 每个阶段通道数
                 layer_planes=(384, 768, 1536),
                 # 每个阶段块个数
                 layer_blocks=(4, 8, 4),
                 # 是否执行空间下采样
                 down_samples=(1, 1, 1),
                 # 是否对第一个1x1逐点卷积执行分组操作
                 with_groups=(0, 1, 1),
                 # 块类型
                 block_layer=None,
                 # 卷积层类型
                 conv_layer=None,
                 # 归一化层类型
                 norm_layer=None,
                 # 激活层类型
                 act_layer=None,
                 ):
        super(ShuffleNetV1Backbone, self).__init__()

        if block_layer is None:
            block_layer = ShuffleNetV1Unit
        if conv_layer is None:
            conv_layer = nn.Conv2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU

        self.stem = make_stem(in_planes,
                              base_planes,
                              conv_layer,
                              norm_layer,
                              act_layer
                              )

        in_planes = base_planes
        for i in range(len(layer_blocks)):
            res_layer = make_stage(in_planes,
                                   layer_planes[i],
                                   groups,
                                   layer_blocks[i],
                                   down_samples[i],
                                   with_groups[i],
                                   block_layer,
                                   conv_layer,
                                   norm_layer,
                                   act_layer
                                   )
            in_planes = layer_planes[i]
            layer_name = f'stage{i + 1}'
            self.add_module(layer_name, res_layer)
        init_weights(self.modules())

    def forward(self, x):
        x = self.stem(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        return x

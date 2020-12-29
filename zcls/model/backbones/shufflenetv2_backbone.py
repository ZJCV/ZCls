# -*- coding: utf-8 -*-

"""
@date: 2020/12/3 下午8:52
@file: mobilenetv2_backbone.py
@author: zj
@description: 
"""
from abc import ABC

import torch.nn as nn
from .shufflenetv2_unit import ShuffleNetV2Unit


def make_stage(  # 输入通道数
        in_planes,
        # 输出通道数
        out_planes,
        # 块个数
        block_num,
        # 是否执行空间下采样
        with_down_sample,
        # 块类型
        block_layer,
        # 卷积层类型
        conv_layer,
        # 归一化层类型
        norm_layer,
        # 激活层类型
        act_layer,
):
    branch_planes = out_planes // 2

    stride = 2 if with_down_sample else 1
    if with_down_sample:
        down_sample = nn.Sequential(
            conv_layer(in_planes, branch_planes, kernel_size=3, stride=stride, padding=1, bias=False),
            norm_layer(branch_planes),
            conv_layer(branch_planes, branch_planes, kernel_size=1, stride=1, padding=0, bias=False),
            norm_layer(branch_planes),
            act_layer(inplace=True)
        )
    else:
        down_sample = None

    blocks = list()
    blocks.append(block_layer(
        in_planes, branch_planes, stride, down_sample, conv_layer, norm_layer, act_layer))
    in_planes = branch_planes

    stride = 1
    down_sample = None
    for i in range(1, block_num):
        blocks.append(block_layer(
            in_planes, branch_planes, stride, down_sample, conv_layer, norm_layer, act_layer))
    return nn.Sequential(*blocks)


class ShuffleNetV2Backbone(nn.Module, ABC):

    def __init__(self,
                 # 输入通道数
                 in_planes=3,
                 # 第一个卷积层通道数
                 base_planes=24,
                 # 输出通道数
                 out_planes=1024,
                 # 每一层通道数
                 layer_planes=(116, 232, 464),
                 # 每一层块个数
                 layer_blocks=(4, 8, 4),
                 # 是否执行空间下采样
                 down_samples=(1, 1, 1),
                 # 块类型
                 block_layer=None,
                 # 卷积层类型
                 conv_layer=None,
                 # 归一化层类型
                 norm_layer=None,
                 # 激活层类型
                 act_layer=None
                 ):
        super(ShuffleNetV2Backbone, self).__init__()

        if block_layer is None:
            block_layer = ShuffleNetV2Unit
        if conv_layer is None:
            conv_layer = nn.Conv2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU

        self._make_stem(in_planes,
                        base_planes,
                        layer_planes[-1],
                        out_planes,
                        conv_layer,
                        norm_layer,
                        act_layer
                        )

        in_planes = base_planes
        for i in range(len(layer_blocks)):
            res_layer = make_stage(in_planes,
                                   layer_planes[i],
                                   layer_blocks[i],
                                   down_samples[i],
                                   block_layer,
                                   conv_layer,
                                   norm_layer,
                                   act_layer
                                   )
            in_planes = layer_planes[i]
            layer_name = f'stage{i + 1}'
            self.add_module(layer_name, res_layer)
        self.init_weights()

    def _make_stem(self,
                   conv1_in_planes,
                   conv1_out_planes,
                   conv5_in_planes,
                   conv5_out_planes,
                   conv_layer,
                   norm_layer,
                   act_layer
                   ):
        self.conv1 = conv_layer(conv1_in_planes, conv1_out_planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm1 = norm_layer(conv1_out_planes)
        self.act = act_layer(inplace=True)

        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

        self.conv5 = conv_layer(conv5_in_planes, conv5_out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm5 = norm_layer(conv5_out_planes)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.pool(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.conv5(x)
        x = self.norm5(x)
        x = self.act(x)

        return x

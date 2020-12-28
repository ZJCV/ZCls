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


class ShuffleNetV1Backbone(nn.Module, ABC):

    def __init__(self,
                 # 输入通道数
                 inplanes=3,
                 # 第一个卷积层通道数
                 base_channel=24,
                 # 分组数
                 groups=8,
                 # 每一层通道数
                 layer_planes=(384, 768, 1536),
                 # 每一层块个数
                 layer_blocks=(4, 8, 4),
                 # 是否执行空间下采样
                 downsamples=(1, 1, 1),
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

        # 输入通道数
        self.inplanes = inplanes
        # 第一个卷积层通道数
        self.base_channel = base_channel
        # 分组数
        self.groups = groups
        # 每一层通道数
        self.layer_planes = layer_planes
        # 每一层块个数
        self.layer_blocks = layer_blocks
        # 是否执行空间下采样
        self.downsamples = downsamples
        # 是否对第一个1x1逐点卷积执行分组操作
        self.with_groups = with_groups
        # 块类型
        self.block_layer = block_layer
        # 卷积层类型
        self.conv_layer = conv_layer
        # 归一化层类型
        self.norm_layer = norm_layer
        # 激活层类型
        self.act_layer = act_layer

        self._make_stem(self.inplanes, self.base_channel)

        self.inplanes = self.base_channel
        self.stage_num = len(self.layer_blocks)
        for i in range(self.stage_num):
            res_layer = self._make_stage(self.inplanes,
                                         self.layer_planes[i],
                                         self.groups,
                                         self.layer_blocks[i],
                                         self.downsamples[i],
                                         self.with_groups[i],
                                         self.block_layer,
                                         self.conv_layer,
                                         self.norm_layer,
                                         self.act_layer
                                         )
            self.inplanes = self.layer_planes[i]
            layer_name = f'stage{i + 1}'
            self.add_module(layer_name, res_layer)
        self._init_weights()

    def _make_stem(self, inplanes, planes):
        self.conv1 = self.conv_layer(inplanes, planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm1 = self.norm_layer(planes)
        self.act = self.act_layer(inplace=True)

        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

    def _make_stage(self,
                    # 输入通道数
                    inplanes,
                    # 输出通道数
                    planes,
                    # 分组数
                    groups,
                    # 块个数
                    block_num,
                    # 是否执行空间下采样
                    with_downsample,
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
        stride = 2 if with_downsample else 1
        downsample = nn.AvgPool2d(3, stride=stride, padding=1) if stride == 2 else None

        blocks = list()
        blocks.append(block_layer(
            inplanes, planes, groups, stride, downsample, with_groups[0], conv_layer, norm_layer, act_layer))
        inplanes = planes

        for i in range(1, block_num):
            blocks.append(block_layer(
                inplanes, planes, 1, 1, None, with_groups[i], conv_layer, norm_layer, act_layer))
        return nn.Sequential(*blocks)

    def _init_weights(self):
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

        for i in range(self.stage_num):
            stage = self.__getattr__(f'stage{i + 1}')
            x = stage(x)

        return x

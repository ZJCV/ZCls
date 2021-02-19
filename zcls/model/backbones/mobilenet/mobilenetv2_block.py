# -*- coding: utf-8 -*-

"""
@date: 2020/12/4 上午11:39
@file: mobilenetv2_block.py
@author: zj
@description: 
"""
from abc import ABC

import torch.nn as nn

from .mobilenetv2_inverted_residual import MobileNetV2InvertedResidual


class MobileNetV2Block(nn.Module, ABC):

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion_rate=1,
                 repeat=1,
                 stride=1,
                 padding=1,
                 conv_layer=None,
                 norm_layer=None,
                 act_layer=None,
                 ):
        """
        repeat InvertedResidualUnit. if stride>1, make downsample to the first Unit
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param expansion_rate: 膨胀因子
        :param repeat: 重复次数
        :param stride: 卷积层步长
        :param padding: 卷积层零填充
        :param conv_layer: 卷积层类型
        :param norm_layer: 归一化层类型
        :param act_layer: 激活层类型
        """
        super(MobileNetV2Block, self).__init__()

        features = list()
        for i in range(repeat):
            if i != 0:
                # 仅对第一个残差块执行下采样操作（如果有的话）
                stride = 1
            features.append(MobileNetV2InvertedResidual(in_channels,
                                                        out_channels,
                                                        expansion_rate=expansion_rate,
                                                        stride=stride,
                                                        padding=padding,
                                                        conv_layer=conv_layer,
                                                        norm_layer=norm_layer,
                                                        act_layer=act_layer))
            in_channels = out_channels

        self.conv = nn.Sequential(*features)

    def forward(self, x):
        return self.conv(x)

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
    """
    重复执行多个反向残差块，每个反向残差块拥有相同的膨胀率和通道数，其中，仅对第一个残差块执行下采样操作（如果有的话）
    """

    def __init__(self,
                 # 输入通道数
                 in_planes,
                 # 输出通道数
                 out_planes,
                 # 膨胀因子
                 expansion_rate=1,
                 # 重复次数
                 repeat=1,
                 # 卷积层步长
                 stride=1,
                 # 卷积层零填充
                 padding=1,
                 # 卷积层类型
                 conv_layer=None,
                 # 归一化层类型
                 norm_layer=None,
                 # 激活层类型
                 act_layer=None,
                 ):
        super(MobileNetV2Block, self).__init__()

        features = list()
        for i in range(repeat):
            if i != 0:
                # 仅对第一个残差块执行下采样操作（如果有的话）
                stride = 1
            features.append(MobileNetV2InvertedResidual(in_planes,
                                                        out_planes,
                                                        expansion_rate=expansion_rate,
                                                        stride=stride,
                                                        padding=padding,
                                                        conv_layer=conv_layer,
                                                        norm_layer=norm_layer,
                                                        act_layer=act_layer))
            in_planes = out_planes

        self.conv = nn.Sequential(*features)

    def forward(self, x):
        return self.conv(x)

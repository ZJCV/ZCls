# -*- coding: utf-8 -*-

"""
@date: 2021/6/3 下午8:24
@file: ghost_module.py
@author: zj
@description: 
"""

import math
import torch
import torch.nn as nn


class GhostModule(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 ratio=2,
                 primary_kernel_size=1,
                 cheap_kernel_size=3,
                 stride=1,
                 is_act=True,
                 ) -> None:
        """
        Block = primary_conv + cheap_operation
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param ratio: 创建内在特征图比率
        :param primary_kernel_size: primary卷积核大小
        :param cheap_kernel_size: cheap_operation深度卷积核大小
        :param stride: primary步长
        :param is_act: 是否执行激活函数
        """
        super(GhostModule, self).__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        primary_padding = primary_kernel_size // 2
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size=(primary_kernel_size, primary_kernel_size),
                      stride=(stride, stride), padding=(primary_padding, primary_padding), bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if is_act else nn.Sequential(),
        )

        cheap_padding = cheap_kernel_size // 2
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, kernel_size=(cheap_kernel_size, cheap_kernel_size), stride=(1, 1),
                      padding=(cheap_padding, cheap_padding), groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if is_act else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]

# -*- coding: utf-8 -*-

"""
@date: 2021/1/4 上午11:32
@file: split_attention_conv2d.py
@author: zj
@description: 
"""
from abc import ABC

import torch

import torch.nn as nn
import torch.nn.functional as F


class SplitAttentionConv2d(nn.Module, ABC):
    """
    ResNetSt的SplitAttention实现，参考：
    1. https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/backbones/resnest.py
    2. https://github.com/zhanghang1989/ResNeSt/blob/73b43ba63d1034dbf3e96b3010a8f2eb4cc3854f/resnest/torch/splat.py
    部分参考./selective_kernel_conv2d.py实现
    """

    def __init__(self,
                 # 输入通道数
                 in_channels,
                 # 输出通道数
                 out_channels,
                 # 每个group中的分离数
                 radix=2,
                 # cardinality
                 groups=1,
                 # 中间层衰减率
                 reduction_rate=16,
                 # 默认中间层最小通道数
                 default_channels: int = 32,
                 # 维度
                 dimension: int = 2
                 ):
        super(SplitAttentionConv2d, self).__init__()

        # split
        self.split = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * radix, kernel_size=3, stride=1, padding=1, bias=False,
                      groups=groups * radix),
            nn.BatchNorm2d(out_channels * radix),
            nn.ReLU(inplace=True)
        )
        # fuse
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        inner_channels = max(out_channels // reduction_rate, default_channels)
        self.compact = nn.Sequential(
            nn.Conv2d(out_channels, inner_channels, kernel_size=1, stride=1, padding=0, bias=False,
                      groups=groups),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True)
        )
        # select
        self.select = nn.Conv2d(inner_channels, out_channels * radix, kernel_size=1, stride=1, bias=False,
                                groups=groups)
        self.softmax = nn.Softmax(dim=0)
        self.dimension = dimension
        self.out_channels = out_channels
        self.radix = radix

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        N, C, H, W = x.shape[:4]

        # split
        out = self.split(x)
        split_out = torch.stack(torch.split(out, self.out_channels, dim=1))
        # fuse
        u = torch.sum(split_out, dim=0)
        s = self.pool(u)
        z = self.compact(s)
        # select
        c = self.select(z)
        split_c = torch.stack(torch.split(c, self.out_channels, dim=1))
        softmax_c = self.softmax(split_c)

        v = torch.sum(split_out.mul(softmax_c), dim=0)
        return v.contiguous()

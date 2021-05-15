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

from ..init_helper import init_weights


class SplitAttentionConv2d(nn.Module, ABC):

    def __init__(self,
                 in_channels,
                 out_channels,
                 radix=2,
                 groups=1,
                 reduction_rate=4,
                 default_channels: int = 32,
                 dimension: int = 2
                 ):
        """
        Implementation of SplitAttention in ResNetSt, refer to
        1. https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/backbones/resnest.py
        2. https://github.com/zhanghang1989/ResNeSt/blob/73b43ba63d1034dbf3e96b3010a8f2eb4cc3854f/resnest/torch/splat.py
        Partial reference ./selective_kernel_conv2d.py implementation
        :param in_channels:
        :param out_channels:
        :param radix:
        :param groups:
        :param reduction_rate:
        :param default_channels:
        :param dimension:
        """
        super(SplitAttentionConv2d, self).__init__()

        # split
        self.split = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * radix, kernel_size=3, stride=1, padding=1, bias=False,
                      groups=groups * radix),
            nn.BatchNorm2d(out_channels * radix),
            nn.ReLU(inplace=True)
        )
        # self.conv1 = nn.Conv2d(in_channels, out_channels * radix, kernel_size=3, stride=1, padding=1, bias=False,
        #                        groups=groups * radix)
        # self.bn1 = nn.BatchNorm2d(out_channels * radix)
        # self.relu = nn.ReLU(inplace=True)
        # fuse
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        inner_channels = max(in_channels * radix // reduction_rate, 32)
        # inner_channels = max(out_channels // reduction_rate, default_channels)
        self.compact = nn.Sequential(
            nn.Conv2d(out_channels, inner_channels, kernel_size=1, stride=1, padding=0, bias=True,
                      groups=groups),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True)
        )
        # select
        self.select = nn.Conv2d(inner_channels, out_channels * radix, kernel_size=1, stride=1,
                                groups=groups)
        # self.softmax = nn.Softmax(dim=0)
        self.rsoftmax = rSoftMax(radix, groups)
        self.dimension = dimension
        self.out_channels = out_channels
        self.radix = radix
        self.groups = groups

        init_weights(self.modules())

    def forward(self, x):
        N, C, H, W = x.shape[:4]
        # split
        out = self.split(x)
        # out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)
        split_out = torch.stack(torch.split(out, self.out_channels, dim=1))
        # fuse
        u = torch.sum(split_out, dim=0)
        s = self.pool(u)
        z = self.compact(s)
        # select
        c = self.select(z)
        softmax_c = self.rsoftmax(c).view(N, -1, 1, 1)

        attens = torch.split(softmax_c, self.out_channels, dim=1)
        v = sum([att * split for (att, split) in zip(attens, split_out)])
        return v.contiguous()
        # split_c = torch.stack(torch.split(c, self.out_channels, dim=1))
        # softmax_c = self.softmax(split_c)
        #
        # v = torch.sum(split_out.mul(softmax_c), dim=0)
        # return v.contiguous()


class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x

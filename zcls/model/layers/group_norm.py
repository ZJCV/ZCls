# -*- coding: utf-8 -*-

"""
@date: 2020/11/25 下午8:44
@file: group_norm.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn


class GroupNorm(nn.Module):
    """
    refer to [group_normalization/group_norm.py](https://github.com/taokong/group_normalization/blob/master/group_norm.py)
    """

    def __init__(self, num_groups: int = 4, num_channels: int = 32, eps: float = 1e-5, affine: bool = True):
        super(GroupNorm, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = torch.ones(num_channels, 1, 1)
            self.bias = torch.zeros(num_channels, 1, 1)

    def forward(self, inputs):
        shape = inputs.size()
        N, C = shape[:2]

        x = inputs.view(N, self.num_groups, -1)

        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)

        x = (x - mean) / (std + self.eps)
        x = x.view(shape)

        return x * self.weight + self.bias

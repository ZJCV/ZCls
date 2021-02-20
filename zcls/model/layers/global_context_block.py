# -*- coding: utf-8 -*-

"""
@date: 2020/12/16 下午7:52
@file: global_context_block.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn


class _GlobalContextBlockND(nn.Module):
    """
    refer to [context_block.py](https://github.com/xvjiarui/GCNet/blob/master/mmdet/ops/gcb/context_block.py)
    """

    def __init__(self,
                 in_channels,
                 reduction=16,
                 dimension=2
                 ):
        """
        :param in_channels:
        :param reduction:
        :param dimension:
        """
        super(_GlobalContextBlockND, self).__init__()
        assert dimension in [1, 2, 3]

        self.in_channels = in_channels
        self.reduction = reduction
        self.dimension = dimension

        if self.dimension == 1:
            self.conv_layer = nn.Conv1d
        elif self.dimension == 2:
            self.conv_layer = nn.Conv2d
        else:
            self.conv_layer = nn.Conv3d

        self._construct_gc()
        self.init_weights()

    def _construct_gc(self):
        self.w_k = self.conv_layer(self.in_channels, 1, kernel_size=1, stride=1, padding=0)

        reduction_channel = self.in_channels // self.reduction
        self.w_v1 = self.conv_layer(self.in_channels, reduction_channel, kernel_size=1, stride=1, padding=0)
        self.ln = nn.LayerNorm([reduction_channel, *([1] * self.dimension)])
        self.relu = nn.ReLU(inplace=True)
        self.w_v2 = self.conv_layer(reduction_channel, self.in_channels, kernel_size=1, stride=1, padding=0)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        """
        :param x: (N, C, **)
        """
        self._check_input_dim(x)

        identity = x
        N, C = x.shape[:2]

        # (N, C, **) -> (N, C, D_j) -> (N, 1, C, D_j)
        input_x = x.view(N, C, -1).unsqueeze(1)
        # (N, C, **) -> (N, 1, **) -> (N, 1, D_j)
        context_mask = self.w_k(x).view(N, 1, -1)
        # (N, 1, D_j) -> (N, 1, D_j, 1)
        context_mask = nn.functional.softmax(context_mask, dim=2).unsqueeze(-1)
        # (N, 1, C, D_j) * (N, 1, D_j, 1) -> (N, 1, C, 1) -> (N, C) -> (N, C, **)
        context = torch.matmul(input_x, context_mask).reshape(N, C).reshape(N, C, *([1] * self.dimension))

        out = self.w_v1(context)
        out = self.ln(out)
        out = self.relu(out)
        transform = self.w_v2(out)

        z = transform + identity
        return z

    def _check_input_dim(self, input):
        raise NotImplementedError


class GlobalContextBlock1D(_GlobalContextBlockND):

    def __init__(self, in_channels, reduction=16, dimension=1):
        super().__init__(in_channels, reduction, dimension)

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))


class GlobalContextBlock2D(_GlobalContextBlockND):

    def __init__(self, in_channels, reduction=16, dimension=2):
        super().__init__(in_channels, reduction, dimension)

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


class GlobalContextBlock3D(_GlobalContextBlockND):

    def __init__(self, in_channels, reduction=16, dimension=3):
        super().__init__(in_channels, reduction, dimension)

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))

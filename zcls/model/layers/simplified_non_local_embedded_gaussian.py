# -*- coding: utf-8 -*-

"""
@date: 2020/12/8 下午3:08
@file: non_local_embedded_gaussian.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn


class _SimplifiedNonLocalNDEmbeddedGaussian(nn.Module):

    def __init__(self,
                 in_channels,
                 dimension=2,
                 norm_layer=None,
                 zero_init_final_norm=True
                 ):
        """
        refer to [context_block.py](https://github.com/xvjiarui/GCNet/blob/master/mmdet/ops/gcb/context_block.py)
        :param in_channels: Input channels
        :param dimension: Data dimension
        :param norm_layer: Normalized layer type
        :param zero_init_final_norm: Initialize the last BN layer at zero to ensure the consistent connection of the initial model
        """
        super(_SimplifiedNonLocalNDEmbeddedGaussian, self).__init__()
        assert dimension in [1, 2, 3]

        self.in_channels = in_channels
        self.dimension = dimension
        self.norm_layer = norm_layer
        self.zero_init_final_norm = zero_init_final_norm

        if self.dimension == 1:
            self.conv_layer = nn.Conv1d
            if self.norm_layer is None:
                self.norm_layer = nn.BatchNorm1d
        elif self.dimension == 2:
            self.conv_layer = nn.Conv2d
            if self.norm_layer is None:
                self.norm_layer = nn.BatchNorm2d
        else:
            self.conv_layer = nn.Conv3d
            if self.norm_layer is None:
                self.norm_layer = nn.BatchNorm3d

        self._construct_nonlocal()
        self.init_weights()

    def _construct_nonlocal(self):
        self.w_k = self.conv_layer(self.in_channels, 1, kernel_size=1, stride=1, padding=0)
        self.w_v = nn.Sequential(
            self.conv_layer(self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0),
            self.norm_layer(self.in_channels)
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
                if self.zero_init_final_norm:
                    nn.init.constant_(m.weight, 0)
                else:
                    nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        :param x: (N, C, **)
        """
        self._check_input_dim(x)

        identity = x
        N, C = x.shape[:2]

        # (N, C, **) -> (N, C, D_j) -> (N, 1, C, D_j)
        input_x = x.view(N, C, -1).unsqueeze(1)
        # (N, C, **) -> (N, 1, **) -> (N, 1, D_j) -> (N, 1, D_j, 1)
        context_mask = self.w_k(x).view(N, 1, -1).unsqueeze(-1)
        context_mask = nn.functional.softmax(context_mask, dim=2)
        # (N, 1, C, D_j) * (N, 1, D_j, 1) -> (N, 1, C, 1) -> (N, C) -> (N, C, **)
        context = torch.matmul(input_x, context_mask).reshape(N, C).reshape(N, C, *([1] * self.dimension))

        transform = self.w_v(context)

        z = transform + identity
        return z

    def _check_input_dim(self, input):
        raise NotImplementedError


class SimplifiedNonLocal1DEmbeddedGaussian(_SimplifiedNonLocalNDEmbeddedGaussian):

    def __init__(self, in_channels, dimension=1, norm_layer=None, zero_init_final_norm=True):
        super().__init__(in_channels, dimension, norm_layer, zero_init_final_norm)

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))


class SimplifiedNonLocal2DEmbeddedGaussian(_SimplifiedNonLocalNDEmbeddedGaussian):

    def __init__(self, in_channels, dimension=2, norm_layer=None, zero_init_final_norm=True):
        super().__init__(in_channels, dimension, norm_layer, zero_init_final_norm)

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


class SimplifiedNonLocal3DEmbeddedGaussian(_SimplifiedNonLocalNDEmbeddedGaussian):

    def __init__(self, in_channels, dimension=3, norm_layer=None, zero_init_final_norm=True):
        super().__init__(in_channels, dimension, norm_layer, zero_init_final_norm)

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))

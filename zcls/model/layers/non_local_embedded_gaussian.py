# -*- coding: utf-8 -*-

"""
@date: 2020/12/8 下午3:08
@file: non_local_embedded_gaussian.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn


class _NonLocalNDEmbeddedGaussian(nn.Module):
    """
    refer to
    1. https://github.com/AlexHex7/Non-local_pytorch/blob/master/lib/non_local_embedded_gaussian.py
    2. https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/nonlocal_helper.py
    """

    def __init__(self,
                 # 输入通道数
                 in_channels,
                 # 中间卷积层输出通道数
                 inner_channels=None,
                 # 数据维度
                 dimension=2,
                 # 减少成对计算，作用于phi和g
                 with_pool=True,
                 # 归一化层类型
                 norm_layer=None,
                 # 零初始化最后的BN层，保证最开始模型的一致性连接
                 zero_init_final_norm=True
                 ):
        super(_NonLocalNDEmbeddedGaussian, self).__init__()
        assert dimension in [1, 2, 3]

        self.in_channels = in_channels
        self.inner_channels = inner_channels
        self.dimension = dimension
        self.with_pool = with_pool
        self.norm_layer = norm_layer
        self.zero_init_final_norm = zero_init_final_norm

        if self.inner_channels is None:
            self.inner_channels = self.in_channels // 2
            if self.inner_channels == 0:
                self.inner_channels = 1

        if self.dimension == 1:
            self.conv_layer = nn.Conv1d
            if self.norm_layer is None:
                self.norm_layer = nn.BatchNorm1d
            if with_pool:
                self.pool = nn.MaxPool1d(kernel_size=(2))
        elif self.dimension == 2:
            self.conv_layer = nn.Conv2d
            if self.norm_layer is None:
                self.norm_layer = nn.BatchNorm2d
            if with_pool:
                self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        else:
            self.conv_layer = nn.Conv3d
            if self.norm_layer is None:
                self.norm_layer = nn.BatchNorm3d
            if with_pool:
                # 仅对空间维度进行下采样
                self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self._construct_nonlocal()
        self.init_weights()

    def _construct_nonlocal(self):
        self.w_theta = self.conv_layer(self.in_channels, self.inner_channels, kernel_size=1, stride=1, padding=0)
        self.w_phi = self.conv_layer(self.in_channels, self.inner_channels, kernel_size=1, stride=1, padding=0)
        self.w_g = self.conv_layer(self.in_channels, self.inner_channels, kernel_size=1, stride=1, padding=0)
        if self.with_pool:
            self.w_phi = nn.Sequential(
                self.w_phi,
                self.pool
            )
            self.w_g = nn.Sequential(
                self.w_g,
                self.pool
            )

        self.w_z = nn.Sequential(
            self.conv_layer(self.inner_channels, self.in_channels, kernel_size=1, stride=1, padding=0),
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

        # (N, C_in, D_j) -> (N, C_inner, D_j)
        theta_x = self.w_theta(x).view(N, self.inner_channels, -1)
        # (N, C_in, D_j) -> (N, C_inner, D2_j)
        phi_x = self.w_phi(x).view(N, self.inner_channels, -1)
        # (N, C, D_j) * (N, C, D2_j) => (N, D_j, D2_j).
        f = torch.einsum("nct,ncp->ntp", (theta_x, phi_x))

        f_div_C = f * (self.inner_channels ** -0.5)
        # 沿着维度j计算softmax
        f_div_C = nn.functional.softmax(f_div_C, dim=2)

        # (N, C_in, D_j) -> (N, C_inner, D2_j)
        g_x = self.w_g(x).view(N, self.inner_channels, -1)

        # (N, D_j, D2_j) * (N, C_inner, D2_j) => (N, C_inner, D_j).
        y = torch.einsum("ntg,ncg->nct", (f_div_C, g_x))
        # (N, C, D_j) => (N, C, **).
        y = y.view(N, self.inner_channels, *x.shape[2:])

        w_y = self.w_z(y).contiguous()
        z = w_y + identity

        return z

    def _check_input_dim(self, input):
        raise NotImplementedError


class NonLocal1DEmbeddedGaussian(_NonLocalNDEmbeddedGaussian):

    def __init__(self, in_channels, inner_channels=None, dimension=1, with_pool=True, norm_layer=None,
                 zero_init_final_norm=True):
        super().__init__(in_channels, inner_channels, dimension, with_pool, norm_layer, zero_init_final_norm)

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))


class NonLocal2DEmbeddedGaussian(_NonLocalNDEmbeddedGaussian):

    def __init__(self, in_channels, inner_channels=None, dimension=2, with_pool=True, norm_layer=None,
                 zero_init_final_norm=True):
        super().__init__(in_channels, inner_channels, dimension, with_pool, norm_layer, zero_init_final_norm)

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


class NonLocal3DEmbeddedGaussian(_NonLocalNDEmbeddedGaussian):

    def __init__(self, in_channels, inner_channels=None, dimension=3, with_pool=True, norm_layer=None,
                 zero_init_final_norm=True):
        super().__init__(in_channels, inner_channels, dimension, with_pool, norm_layer, zero_init_final_norm)

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))

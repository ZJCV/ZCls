# -*- coding: utf-8 -*-

"""
@date: 2021/2/1 下午7:16
@file: acb_helper.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn
from .layers.asymmetric_convolution_block import AsymmetricConvolutionBlock


def insert_acblock(model: nn.Module):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1:
            # 将标准卷积替换为ACBlock
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size
            stride = module.stride
            padding = module.padding
            dilation = module.dilation
            groups = module.groups
            padding_mode = module.padding_mode

            acblock = AsymmetricConvolutionBlock(in_channels,
                                                 out_channels,
                                                 kernel_size[0],
                                                 stride,
                                                 padding=padding[0],
                                                 padding_mode=padding_mode,
                                                 dilation=dilation,
                                                 groups=groups)
            model.add_module(name, acblock)
        else:
            insert_acblock(module)


def fuse_acblock(model: nn.Module, eps=1e-5):
    for name, module in model.named_children():
        if isinstance(module, AsymmetricConvolutionBlock):
            # 将ACBlock替换为标准卷积
            # 获取NxN卷积的权重以及对应BN的权重、偏置、运行时均值、运行时方差
            square_conv_weight = module.square_conv.weight
            square_bn_weight = module.square_bn.weight
            square_bn_bias = module.square_bn.bias
            square_bn_running_mean = module.square_bn.running_mean
            square_bn_running_std = torch.sqrt(module.square_bn.running_var + eps)
            # 获取1xN卷积的权重以及对应BN的权重、偏置、运行时均值、运行时方差
            vertical_conv_weight = module.ver_conv.weight
            vertical_bn_weight = module.ver_bn.weight
            vertical_bn_bias = module.ver_bn.bias
            vertical_bn_running_mean = module.ver_bn.running_mean
            vertical_bn_running_std = torch.sqrt(module.ver_bn.running_var + eps)
            # 获取1xN卷积的权重以及对应BN的权重、偏置、运行时均值、运行时方差
            horizontal_conv_weight = module.hor_conv.weight
            horizontal_bn_weight = module.hor_bn.weight
            horizontal_bn_bias = module.hor_bn.bias
            horizontal_bn_running_mean = module.hor_bn.running_mean
            horizontal_bn_running_std = torch.sqrt(module.hor_bn.running_var + eps)
            # 计算偏差
            fused_bias = square_bn_bias + vertical_bn_bias + horizontal_bn_bias \
                         - square_bn_running_mean * square_bn_weight / square_bn_running_std \
                         - vertical_bn_running_mean * vertical_bn_weight / vertical_bn_running_std \
                         - horizontal_bn_running_mean * horizontal_bn_weight / horizontal_bn_running_std
            # 计算权重
            fused_kernel = _fuse_kernel(square_conv_weight, square_bn_weight, square_bn_running_std)
            _add_to_square_kernel(fused_kernel,
                                  _fuse_kernel(vertical_conv_weight, vertical_bn_weight, vertical_bn_running_std))
            _add_to_square_kernel(fused_kernel,
                                  _fuse_kernel(horizontal_conv_weight, horizontal_bn_weight, horizontal_bn_running_std))
            # 新建标准卷积，赋值权重和偏差后重新插入模型
            fused_conv = nn.Conv2d(module.in_channels,
                                   module.out_channels,
                                   module.kernel_size,
                                   stride=module.stride,
                                   padding=module.padding,
                                   dilation=module.dilation,
                                   groups=module.groups,
                                   padding_mode=module.padding_mode
                                   )
            fused_conv.weight = nn.Parameter(fused_kernel)
            fused_conv.bias = nn.Parameter(fused_bias)
            model.add_module(name, fused_conv)
        else:
            fuse_acblock(module, eps=eps)


def _fuse_kernel(kernel, gamma, std):
    # b_gamma = np.reshape(gamma, (kernel.shape[0], 1, 1, 1))
    # b_gamma = np.tile(b_gamma, (1, kernel.shape[1], kernel.shape[2], kernel.shape[3]))
    # b_std = np.reshape(std, (kernel.shape[0], 1, 1, 1))
    # b_std = np.tile(b_std, (1, kernel.shape[1], kernel.shape[2], kernel.shape[3]))
    b_gamma = torch.reshape(gamma, (kernel.shape[0], 1, 1, 1))
    b_gamma = b_gamma.repeat(1, kernel.shape[1], kernel.shape[2], kernel.shape[3])
    b_std = torch.reshape(std, (kernel.shape[0], 1, 1, 1))
    b_std = b_std.repeat(1, kernel.shape[1], kernel.shape[2], kernel.shape[3])
    return kernel * b_gamma / b_std


def _add_to_square_kernel(square_kernel, asym_kernel):
    asym_h = asym_kernel.shape[2]
    asym_w = asym_kernel.shape[3]
    square_h = square_kernel.shape[2]
    square_w = square_kernel.shape[3]
    square_kernel[:, :, square_h // 2 - asym_h // 2: square_h // 2 - asym_h // 2 + asym_h,
    square_w // 2 - asym_w // 2: square_w // 2 - asym_w // 2 + asym_w] += asym_kernel

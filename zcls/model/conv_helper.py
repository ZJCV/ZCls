# -*- coding: utf-8 -*-

"""
@date: 2020/12/4 下午4:11
@file: act_helper.py
@author: zj
@description: 
"""

import numpy as np
import torch
import torch.nn as nn
from .layers.asymmetric_convolution_block import AsymmetricConvolutionBlock
from .layers.acb_util import _fuse_kernel, _add_to_square_kernel
from .layers.repvgg_block import RepVGGBlock
from .layers.repvgg_util import get_equivalent_kernel_bias


def get_conv(cfg):
    """
    Args:
        cfg (CfgNode): model building configs, details are in the comments of
            the config file.
    Returns:
        nn.Module: the conv layer.
    """
    conv_type = cfg.MODEL.CONV.TYPE
    if conv_type == "Conv2d":
        return nn.Conv2d
    elif conv_type == "Conv3d":
        return nn.Conv3d
    else:
        raise NotImplementedError(
            "Conv type {} is not supported".format(conv_type)
        )


def insert_acblock(model: nn.Module):
    items = list(model.named_children())
    idx = 0
    while idx < len(items):
        name, module = items[idx]
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
                                                 stride[0],
                                                 padding=padding[0],
                                                 padding_mode=padding_mode,
                                                 dilation=dilation,
                                                 groups=groups)
            model.add_module(name, acblock)
            # 如果conv层之后跟随着BN层，那么删除该BN层
            # 参考[About BN layer #35](https://github.com/DingXiaoH/ACNet/issues/35)
            if (idx + 1) < len(items) and isinstance(items[idx + 1][1], nn.BatchNorm2d):
                new_layer = nn.Identity()
                model.add_module(items[idx + 1][0], new_layer)
        else:
            insert_acblock(module)
        idx += 1


def fuse_acblock(model: nn.Module, eps=1e-5):
    for name, module in model.named_children():
        if isinstance(module, AsymmetricConvolutionBlock):
            # 将ACBlock替换为标准卷积
            # 获取Nx1卷积的权重以及对应BN的权重、偏置、运行时均值、运行时方差
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
            if isinstance(module.square_bn, nn.Identity):
                # 获取NxN卷积的权重以及对应BN的权重、偏置、运行时均值、运行时方差
                # 在insert_repvgg_block过程中将Conv后面的BN设置nn.Identity
                square_weight = module.square_conv.weight
                square_bias = module.square_conv.bias
                # 计算偏差
                fused_bias = square_bias + vertical_bn_bias + horizontal_bn_bias \
                             - vertical_bn_running_mean * vertical_bn_weight / vertical_bn_running_std \
                             - horizontal_bn_running_mean * horizontal_bn_weight / horizontal_bn_running_std
                # 计算权重
                fused_kernel = square_weight
            else:
                square_conv_weight = module.square_conv.weight
                square_bn_weight = module.square_bn.weight
                square_bn_bias = module.square_bn.bias
                square_bn_running_mean = module.square_bn.running_mean
                square_bn_running_std = torch.sqrt(module.square_bn.running_var + eps)
                # 计算偏差
                fused_bias = square_bn_bias + vertical_bn_bias + horizontal_bn_bias \
                             - square_bn_running_mean * square_bn_weight / square_bn_running_std \
                             - vertical_bn_running_mean * vertical_bn_weight / vertical_bn_running_std \
                             - horizontal_bn_running_mean * horizontal_bn_weight / horizontal_bn_running_std
                # 计算权重
                fused_kernel = _fuse_kernel(square_conv_weight, square_bn_weight, square_bn_running_std)
            # 计算权重
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
            fused_conv.weight = nn.Parameter(fused_kernel.detach().cpu())
            fused_conv.bias = nn.Parameter(fused_bias.detach().cpu())
            model.add_module(name, fused_conv)
        else:
            fuse_acblock(module, eps=eps)


def insert_repvgg_block(model: nn.Module):
    items = list(model.named_children())
    idx = 0
    while idx < len(items):
        name, module = items[idx]
        if isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1:
            # 将标准卷积替换为RepVGGBlock
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size
            stride = module.stride
            padding = module.padding
            dilation = module.dilation
            groups = module.groups
            padding_mode = module.padding_mode

            acblock = RepVGGBlock(in_channels,
                                  out_channels,
                                  kernel_size[0],
                                  stride[0],
                                  padding=padding[0],
                                  padding_mode=padding_mode,
                                  dilation=dilation,
                                  groups=groups)
            model.add_module(name, acblock)
            # 如果conv层之后跟随着BN层，那么删除该BN层
            # 参考[About BN layer #35](https://github.com/DingXiaoH/ACNet/issues/35)
            if (idx + 1) < len(items) and isinstance(items[idx + 1][1], nn.BatchNorm2d):
                new_layer = nn.Identity()
                model.add_module(items[idx + 1][0], new_layer)
        else:
            insert_repvgg_block(module)
        idx += 1


def fuse_repvgg_block(model: nn.Module):
    for name, module in model.named_children():
        if isinstance(module, RepVGGBlock):
            # 将RepVGGBlock替换为标准卷积
            kernel, bias = get_equivalent_kernel_bias(module.rbr_dense,
                                                      module.rbr_1x1,
                                                      module.rbr_identity,
                                                      module.in_channels,
                                                      module.groups,
                                                      module.padding)
            # 新建标准卷积，赋值权重和偏差后重新插入模型
            fused_conv = nn.Conv2d(module.in_channels,
                                   module.out_channels,
                                   module.kernel_size,
                                   stride=module.stride,
                                   padding=module.padding,
                                   dilation=module.dilation,
                                   groups=module.groups,
                                   padding_mode=module.padding_mode,
                                   bias=True
                                   )
            fused_conv.weight = nn.Parameter(kernel.detach().cpu())
            fused_conv.bias = nn.Parameter(bias.detach().cpu())
            model.add_module(name, fused_conv)
        else:
            fuse_repvgg_block(module)

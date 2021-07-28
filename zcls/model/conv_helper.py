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
from .layers import repvgg_util
from .layers.diverse_branch_block import DiverseBranchBlock
from .layers import dbb_util


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
            # Replace standard convolution with acblock
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
            # If conv layer is followed by BN layer, delete the BN layer
            # refer to [About BN layer #35](https://github.com/DingXiaoH/ACNet/issues/35)
            if (idx + 1) < len(items) and isinstance(items[idx + 1][1], nn.BatchNorm2d):
                new_layer = nn.Identity()
                model.add_module(items[idx + 1][0], new_layer)
        else:
            insert_acblock(module)
        idx += 1


def fuse_acblock(model: nn.Module, eps=1e-5):
    for name, module in model.named_children():
        if isinstance(module, AsymmetricConvolutionBlock):
            # Replace acblock with standard convolution
            # Obtain the weight of NX1 convolution and the weight, offset, runtime mean and runtime variance of corresponding BN
            vertical_conv_weight = module.ver_conv.weight
            vertical_bn_weight = module.ver_bn.weight
            vertical_bn_bias = module.ver_bn.bias
            vertical_bn_running_mean = module.ver_bn.running_mean
            vertical_bn_running_std = torch.sqrt(module.ver_bn.running_var + eps)
            # Obtain the weight of 1xN convolution and the weight, offset, runtime mean and runtime variance of the corresponding BN
            horizontal_conv_weight = module.hor_conv.weight
            horizontal_bn_weight = module.hor_bn.weight
            horizontal_bn_bias = module.hor_bn.bias
            horizontal_bn_running_mean = module.hor_bn.running_mean
            horizontal_bn_running_std = torch.sqrt(module.hor_bn.running_var + eps)
            if isinstance(module.square_bn, nn.Identity):
                square_weight = module.square_conv.weight
                square_bias = module.square_conv.bias
                # Calculation bias
                fused_bias = square_bias + vertical_bn_bias + horizontal_bn_bias \
                             - vertical_bn_running_mean * vertical_bn_weight / vertical_bn_running_std \
                             - horizontal_bn_running_mean * horizontal_bn_weight / horizontal_bn_running_std
                # Calculate weight
                fused_kernel = square_weight
            else:
                square_conv_weight = module.square_conv.weight
                square_bn_weight = module.square_bn.weight
                square_bn_bias = module.square_bn.bias
                square_bn_running_mean = module.square_bn.running_mean
                square_bn_running_std = torch.sqrt(module.square_bn.running_var + eps)
                # Calculate bias
                fused_bias = square_bn_bias + vertical_bn_bias + horizontal_bn_bias \
                             - square_bn_running_mean * square_bn_weight / square_bn_running_std \
                             - vertical_bn_running_mean * vertical_bn_weight / vertical_bn_running_std \
                             - horizontal_bn_running_mean * horizontal_bn_weight / horizontal_bn_running_std
                # Calculate weight
                fused_kernel = _fuse_kernel(square_conv_weight, square_bn_weight, square_bn_running_std)
            # Calculate weight
            _add_to_square_kernel(fused_kernel,
                                  _fuse_kernel(vertical_conv_weight, vertical_bn_weight, vertical_bn_running_std))
            _add_to_square_kernel(fused_kernel,
                                  _fuse_kernel(horizontal_conv_weight, horizontal_bn_weight, horizontal_bn_running_std))
            # Create a new standard convolution, assign weight and bias, and reinsert the model
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
            # Replace the standard convolution with repvgg_block
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
            # If conv layer is followed by BN layer, delete the BN layer
            # refer to [About BN layer #35](https://github.com/DingXiaoH/ACNet/issues/35)
            if (idx + 1) < len(items) and isinstance(items[idx + 1][1], nn.BatchNorm2d):
                new_layer = nn.Identity()
                model.add_module(items[idx + 1][0], new_layer)
        else:
            insert_repvgg_block(module)
        idx += 1


def fuse_repvgg_block(model: nn.Module):
    for name, module in model.named_children():
        if isinstance(module, RepVGGBlock):
            # Replace repvgg_block with standard convolution
            kernel, bias = repvgg_util.get_equivalent_kernel_bias(module.rbr_dense,
                                                                  module.rbr_1x1,
                                                                  module.rbr_identity,
                                                                  module.in_channels,
                                                                  module.groups,
                                                                  module.padding)
            # Create a new standard convolution, assign weight and bias, and reinsert the model
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


def insert_dbblock(model: nn.Module):
    items = list(model.named_children())
    idx = 0
    while idx < len(items):
        name, module = items[idx]
        if isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1:
            # Replace the standard convolution with dbblock
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size
            stride = module.stride
            padding = module.padding
            dilation = module.dilation
            groups = module.groups

            acblock = DiverseBranchBlock(in_channels,
                                         out_channels,
                                         kernel_size[0],
                                         stride[0],
                                         padding=padding[0],
                                         dilation=dilation[0],
                                         groups=groups)
            model.add_module(name, acblock)
            # If conv layer is followed by BN layer, delete the BN layer
            # refer to [About BN layer #35](https://github.com/DingXiaoH/ACNet/issues/35)
            if (idx + 1) < len(items) and isinstance(items[idx + 1][1], nn.BatchNorm2d):
                new_layer = nn.Identity()
                model.add_module(items[idx + 1][0], new_layer)
        else:
            insert_dbblock(module)
        idx += 1


def fuse_dbblock(model: nn.Module):
    for name, module in model.named_children():
        if isinstance(module, DiverseBranchBlock):
            # Replace dbblock with standard convolution
            kernel, bias = dbb_util.get_equivalent_kernel_bias(module)

            # Create a new standard convolution, assign weight and bias, and reinsert the model
            fused_conv = nn.Conv2d(module.in_channels,
                                   module.out_channels,
                                   module.kernel_size,
                                   stride=module.stride,
                                   padding=module.padding,
                                   dilation=module.dilation,
                                   groups=module.groups,
                                   bias=True
                                   )
            fused_conv.weight = nn.Parameter(kernel.detach().cpu())
            fused_conv.bias = nn.Parameter(bias.detach().cpu())
            model.add_module(name, fused_conv)
        else:
            fuse_dbblock(module)

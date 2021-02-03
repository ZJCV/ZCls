# -*- coding: utf-8 -*-

"""
@date: 2021/2/2 下午8:51
@file: repvgg_util.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn
import numpy as np


#   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
#   You can get the equivalent kernel and bias at any time and do whatever you want,
#   for example, apply some penalties or constraints during training, just like you do to the other models.
#   May be useful for quantization or pruning.
def get_equivalent_kernel_bias(rbr_dense, rbr_1x1, rbr_identity, in_channels, groups, padding_11):
    kernel3x3, bias3x3 = _fuse_bn_tensor(rbr_dense, in_channels, groups)
    kernel1x1, bias1x1 = _fuse_bn_tensor(rbr_1x1, in_channels, groups)
    kernelid, biasid = _fuse_bn_tensor(rbr_identity, in_channels, groups)
    return kernel3x3 + _pad_1x1_to_3x3_tensor(kernel1x1, padding_11) + kernelid, bias3x3 + bias1x1 + biasid


def _pad_1x1_to_3x3_tensor(kernel1x1, padding_11=1):
    if kernel1x1 is None:
        return 0
    else:
        # return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])
        return torch.nn.functional.pad(kernel1x1, [padding_11] * 4)


def _fuse_bn_tensor(branch, in_channels, groups):
    if branch is None:
        return 0, 0
    if isinstance(branch, nn.Sequential):
        layer_list = list(branch)
        if len(layer_list) == 2 and isinstance(layer_list[1], nn.Identity):
            # conv/bn已经在acb中进行了融合
            return branch.conv.weight, branch.conv.bias
        kernel = branch.conv.weight
        running_mean = branch.bn.running_mean
        running_var = branch.bn.running_var
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn.eps
    else:
        assert isinstance(branch, nn.BatchNorm2d)
        input_dim = in_channels // groups
        kernel_value = np.zeros((in_channels, input_dim, 3, 3), dtype=np.float32)
        for i in range(in_channels):
            kernel_value[i, i % input_dim, 1, 1] = 1

        kernel = torch.from_numpy(kernel_value).to(branch.weight.device)
        running_mean = branch.running_mean
        running_var = branch.running_var
        gamma = branch.weight
        beta = branch.bias
        eps = branch.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std

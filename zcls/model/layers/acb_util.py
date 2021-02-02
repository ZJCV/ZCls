# -*- coding: utf-8 -*-

"""
@date: 2021/2/2 下午8:53
@file: acb_util.py
@author: zj
@description: 
"""

import torch


def _fuse_kernel(kernel, gamma, std):
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

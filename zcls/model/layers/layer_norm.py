# -*- coding: utf-8 -*-

"""
@date: 2020/11/25 下午8:44
@file: layer_norm.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn


class LayerNorm(nn.LayerNorm):

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=False):
        super(LayerNorm, self).__init__(normalized_shape, eps, elementwise_affine)

    def forward(self, input):
        # calculate running estimates
        mean = input.mean([1, 2, 3])
        # use biased var in train
        var = input.var([1, 2, 3], unbiased=False)

        input = (input - mean[:, None, None, None]) / (torch.sqrt(var[:, None, None, None] + self.eps))
        if self.elementwise_affine:
            input = input * self.weight[None, :] + self.bias[None, :]

        return input

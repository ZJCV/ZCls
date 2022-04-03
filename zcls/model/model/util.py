# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午10:55
@file: util.py
@author: zj
@description: 
"""

import math

import torch.nn as nn


def create_linear(in_features, out_features, bias=True):
    fc = nn.Linear(in_features, out_features, bias=bias)
    reset_linear_parameters(fc)

    return fc


def reset_linear_parameters(module):
    """
    refer to: [[PyTorch]torch.nn各个层使用的默认的初始化分布](https://zhuanlan.zhihu.com/p/190207193)
    """
    assert isinstance(module, nn.Linear)
    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
    if module.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(module.bias, -bound, bound)

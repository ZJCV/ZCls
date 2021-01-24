# -*- coding: utf-8 -*-

"""
@date: 2021/1/1 下午9:19
@file: test_selective_kernel_conv2d.py
@author: zj
@description: 
"""

import numpy as np
import torch
import torch.nn as nn
from resnest.torch.splat import SplAtConv2d

from zcls.model.layers.split_attention_conv2d import SplitAttentionConv2d


def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.constant_(m.weight, 0.01)
            # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            # nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.weight, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def get_custom_res():
    # 保证训练时获取的随机数都是一样的
    init_seed = 1
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed(init_seed)
    np.random.seed(init_seed)

    num = 3
    in_channels = 64
    out_channels = 128
    data = torch.randn(num, in_channels, 56, 56)

    # custom
    model = SplitAttentionConv2d(in_channels,
                                 out_channels,
                                 radix=2,
                                 groups=32,
                                 reduction_rate=16
                                 )
    init_weights(model.modules())
    print(model)
    outputs_c = model(data)
    print(outputs_c.shape)

    return outputs_c


def get_official_res():
    # 保证训练时获取的随机数都是一样的
    init_seed = 1
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed(init_seed)
    np.random.seed(init_seed)

    num = 3
    in_channels = 64
    out_channels = 128
    data = torch.randn(num, in_channels, 56, 56)

    # official
    model = SplAtConv2d(in_channels,
                        out_channels,
                        3,
                        norm_layer=nn.BatchNorm2d,
                        bias=False,
                        padding=1,
                        radix=2,
                        groups=32,
                        reduction_factor=16
                        )
    init_weights(model.modules())
    print(model)
    outputs_o = model(data)
    print(outputs_o.shape)

    return outputs_o


def compare():
    outputs_c = get_custom_res()
    outputs_o = get_official_res()

    res = torch.allclose(outputs_c, outputs_o)
    print(res)


def test_split_attention_conv2d():
    num = 3
    in_channels = 64
    out_channels = 128
    data = torch.randn(num, in_channels, 56, 56)

    # 不进行分组
    model = SplitAttentionConv2d(in_channels,
                                 out_channels,
                                 radix=2,
                                 groups=1,
                                 reduction_rate=4
                                 )
    print(model)
    outputs = model(data)
    print(outputs.shape)

    assert outputs.shape == (num, out_channels, 56, 56)

    # 不进行radix
    model = SplitAttentionConv2d(in_channels,
                                 out_channels,
                                 radix=1,
                                 groups=1,
                                 reduction_rate=16
                                 )
    print(model)
    outputs = model(data)
    print(outputs.shape)

    assert outputs.shape == (num, out_channels, 56, 56)

    # 同时实现radix和group
    model = SplitAttentionConv2d(in_channels,
                                 out_channels,
                                 radix=2,
                                 groups=32,
                                 reduction_rate=16
                                 )
    print(model)
    outputs = model(data)
    print(outputs.shape)

    assert outputs.shape == (num, out_channels, 56, 56)


if __name__ == '__main__':
    compare()
    test_split_attention_conv2d()

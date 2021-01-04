# -*- coding: utf-8 -*-

"""
@date: 2021/1/1 下午9:19
@file: test_selective_kernel_conv2d.py
@author: zj
@description: 
"""

import torch

from zcls.model.layers.split_attention_conv2d import SplitAttentionConv2d


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
                                 reduction_rate=16
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
    test_split_attention_conv2d()

# -*- coding: utf-8 -*-

"""
@date: 2021/1/1 下午9:19
@file: test_selective_kernel_conv2d.py
@author: zj
@description: 
"""

import torch

from zcls.model.layers.selective_kernel_conv2d import SelectiveKernelConv2d


def test_selective_kernel_conv2d():
    num = 3
    in_channels = 64
    out_channels = 256
    data = torch.randn(num, in_channels, 56, 56)

    # 不进行下采样
    model = SelectiveKernelConv2d(in_channels,
                                  out_channels,
                                  stride=1,
                                  groups=32,
                                  reduction_rate=16
                                  )
    print(model)
    outputs = model(data)
    print(outputs.shape)

    assert outputs.shape == (num, out_channels, 56, 56)

    # 进行下采样
    model = SelectiveKernelConv2d(in_channels,
                                  out_channels,
                                  stride=2,
                                  groups=32,
                                  reduction_rate=16
                                  )
    print(model)
    outputs = model(data)
    print(outputs.shape)

    assert outputs.shape == (num, out_channels, 28, 28)


if __name__ == '__main__':
    test_selective_kernel_conv2d()

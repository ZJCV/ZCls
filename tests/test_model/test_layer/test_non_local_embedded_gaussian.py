# -*- coding: utf-8 -*-

"""
@date: 2020/12/14 下午7:29
@file: test_non_local_embedded_gaussian.py
@author: zj
@description: 
"""

import torch

from zcls.model.layers.non_local_embedded_gaussian import NonLocal1DEmbeddedGaussian, \
    NonLocal2DEmbeddedGaussian, NonLocal3DEmbeddedGaussian


def test_non_local_embedded_gaussian_1d():
    N = 10
    C = 128
    inner_channel = C // 16

    data = torch.randn(N, C, 7)
    model = NonLocal1DEmbeddedGaussian(in_channels=C, inner_channels=inner_channel)
    print(model)

    outputs = model(data)
    print(outputs.shape)

    assert outputs.shape == (N, C, 7)


def test_non_local_embedded_gaussian_2d():
    N = 10
    C = 128
    inner_channel = C // 16

    data = torch.randn(N, C, 7, 7)
    model = NonLocal2DEmbeddedGaussian(in_channels=C, inner_channels=inner_channel)
    print(model)

    outputs = model(data)
    print(outputs.shape)

    assert outputs.shape == (N, C, 7, 7)


def test_non_local_embedded_gaussian_3d():
    N = 10
    C = 128
    inner_channel = C // 16

    data = torch.randn(N, C, 4, 7, 7)
    model = NonLocal3DEmbeddedGaussian(in_channels=C, inner_channels=inner_channel)
    print(model)

    outputs = model(data)
    print(outputs.shape)

    assert outputs.shape == (N, C, 4, 7, 7)


if __name__ == '__main__':
    test_non_local_embedded_gaussian_1d()
    test_non_local_embedded_gaussian_2d()
    test_non_local_embedded_gaussian_3d()

# -*- coding: utf-8 -*-

"""
@date: 2020/12/14 下午7:14
@file: test_squeeze_and_excitation_block.py
@author: zj
@description: 
"""

import torch

from zcls.model.layers.squeeze_and_excitation_block import SqueezeAndExcitationBlock1D, \
    SqueezeAndExcitationBlock2D, SqueezeAndExcitationBlock3D


def test_squeeze_and_excitation_block_1d():
    N = 10
    C = 128
    reduction = 16

    data = torch.randn(N, C, 7)
    model = SqueezeAndExcitationBlock1D(in_channels=C, reduction=reduction)
    print(model)

    outputs = model(data)
    print(outputs.shape)

    assert outputs.shape == (N, C, 7)


def test_squeeze_and_excitation_block_2d():
    N = 10
    C = 128
    reduction = 16

    data = torch.randn(N, C, 7, 7)
    model = SqueezeAndExcitationBlock2D(in_channels=C, reduction=reduction)
    print(model)

    outputs = model(data)
    print(outputs.shape)

    assert outputs.shape == (N, C, 7, 7)


def test_squeeze_and_excitation_block_3d():
    N = 10
    C = 128
    reduction = 16

    data = torch.randn(N, C, 4, 7, 7)
    model = SqueezeAndExcitationBlock3D(in_channels=C, reduction=reduction)
    print(model)

    outputs = model(data)
    print(outputs.shape)

    assert outputs.shape == (N, C, 4, 7, 7)


if __name__ == '__main__':
    test_squeeze_and_excitation_block_1d()
    test_squeeze_and_excitation_block_2d()
    test_squeeze_and_excitation_block_3d()

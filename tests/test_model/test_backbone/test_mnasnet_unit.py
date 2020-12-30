# -*- coding: utf-8 -*-

"""
@date: 2020/12/30 下午5:28
@file: test_mnasnet_unit.py
@author: zj
@description: 
"""

import torch

from zcls.model.backbones.mnasnet_unit import MNASNetUint


def test_mnasnet_unit():
    data = torch.randn(1, 32, 112, 112)

    # 测试SepConv(k3x3)
    model = MNASNetUint(32,
                        16,
                        stride=1,
                        kernel_size=3,
                        expansion_rate=1,
                        with_attention=False
                        )
    print(model)
    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 16, 112, 112)

    # 测试MBConv6(k3x3)
    model = MNASNetUint(16,
                        24,
                        stride=2,
                        kernel_size=3,
                        expansion_rate=6,
                        with_attention=False
                        )
    print(model)
    outputs = model(outputs)
    print(outputs.shape)
    assert outputs.shape == (1, 24, 56, 56)

    # 测试MBConv3(k5x5), SE
    model = MNASNetUint(24,
                        40,
                        stride=2,
                        kernel_size=5,
                        expansion_rate=3,
                        with_attention=True
                        )
    print(model)
    outputs = model(outputs)
    print(outputs.shape)
    assert outputs.shape == (1, 40, 28, 28)

    # 测试MBConv6(k3x3), SE
    model = MNASNetUint(40,
                        112,
                        stride=2,
                        kernel_size=3,
                        expansion_rate=6,
                        with_attention=True
                        )
    print(model)
    outputs = model(outputs)
    print(outputs.shape)
    assert outputs.shape == (1, 112, 14, 14)

    # 测试MBConv6(k5x5), SE
    model = MNASNetUint(112,
                        160,
                        stride=2,
                        kernel_size=5,
                        expansion_rate=6,
                        with_attention=True
                        )
    print(model)
    outputs = model(outputs)
    print(outputs.shape)
    assert outputs.shape == (1, 160, 7, 7)


if __name__ == '__main__':
    test_mnasnet_unit()

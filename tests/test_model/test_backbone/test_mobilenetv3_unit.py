# -*- coding: utf-8 -*-

"""
@date: 2020/12/30 下午9:30
@file: test_mobilenetv3_unit.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn

from zcls.model.backbones.mobilenet.mobilenetv3_unit import MobileNetV3Uint


def test_mobilenet_v3_unit():
    # 3x3, 不进行下采样
    data = torch.randn(1, 16, 112, 112)
    model = MobileNetV3Uint(in_channels=16,
                            inner_channels=16,
                            out_channels=16,
                            stride=1,
                            kernel_size=3,
                            with_attention=False,
                            act_layer=nn.ReLU
                            )
    print(model)
    outputs = model(data)
    print(outputs.shape)

    assert outputs.shape == (1, 16, 112, 112)

    # 5x5, 进行下采样
    data = torch.randn(1, 24, 56, 56)
    model = MobileNetV3Uint(in_channels=24,
                            inner_channels=72,
                            out_channels=40,
                            stride=2,
                            kernel_size=5,
                            with_attention=True,
                            reduction=4,
                            attention_type='SqueezeAndExcitationBlock2D'
                            )
    print(model)
    outputs = model(data)
    print(outputs.shape)

    assert outputs.shape == (1, 40, 28, 28)


if __name__ == '__main__':
    test_mobilenet_v3_unit()

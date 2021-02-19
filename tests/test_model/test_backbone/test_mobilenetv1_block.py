# -*- coding: utf-8 -*-

"""
@date: 2020/12/3 下午7:42
@file: test_mobilenetv1_block.py
@author: zj
@description: 
"""

import torch

from zcls.model.backbones.mobilenet.mobilenetv1_block import MobileNetV1Block


def test_mobilenet_v1_block():
    # 不进行下采样
    data = torch.randn(1, 32, 112, 112)
    inplanes = 32
    planes = 64
    stride = 1
    model = MobileNetV1Block(inplanes, planes, stride)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 64, 112, 112)

    # 进行下采样
    data = torch.randn(1, 64, 112, 112)
    inplanes = 64
    planes = 128
    stride = 2
    model = MobileNetV1Block(inplanes, planes, stride)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 128, 56, 56)

if __name__ == '__main__':
    test_mobilenet_v1_block()

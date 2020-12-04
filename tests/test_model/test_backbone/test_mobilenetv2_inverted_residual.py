# -*- coding: utf-8 -*-

"""
@date: 2020/12/4 下午3:21
@file: test_mobilenetv2_inverted_residual.py
@author: zj
@description: 
"""

import torch

from zcls.model.backbones.mobilenetv2_inverted_residual import MobileNetV2InvertedResidual


def test_mobilenetv2_inverted_residual():
    # 不进行下采样、不进行膨胀
    data = torch.randn(1, 32, 112, 112)
    inplanes = 32
    planes = 16
    stride = 1
    model = MobileNetV2InvertedResidual(inplanes,
                                        planes,
                                        stride=stride,
                                        t=1.0)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 16, 112, 112)

    # 进行下采样，进行膨胀
    data = torch.randn(1, 16, 112, 112)
    inplanes = 16
    planes = 24
    stride = 2
    model = MobileNetV2InvertedResidual(inplanes,
                                        planes,
                                        stride=stride,
                                        t=6.0)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 24, 56, 56)

    # 不进行下采样、进行膨胀
    data = torch.randn(1, 24, 56, 56)
    inplanes = 24
    planes = 24
    stride = 1
    model = MobileNetV2InvertedResidual(inplanes,
                                        planes,
                                        stride=stride,
                                        t=6.0)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 24, 56, 56)


if __name__ == '__main__':
    test_mobilenetv2_inverted_residual()

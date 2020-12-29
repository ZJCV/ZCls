# -*- coding: utf-8 -*-

"""
@date: 2020/12/25 上午10:14
@file: test_shufflenetv1_unit.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn
from zcls.model.backbones.shufflenetv1_unit import ShuffleNetV1Unit


def test_shufflenet_v1_unit():
    data = torch.randn(1, 24, 56, 56)
    inplanes = 24
    planes = 384
    groups = 8

    # 下采样，第一个1x1逐点卷积层不进行分组
    stride = 2
    downsample = nn.AvgPool2d(3, stride=stride, padding=1) if stride == 2 else None
    model = ShuffleNetV1Unit(inplanes, planes, groups, stride, downsample, with_group=False)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, planes, 28, 28)

    # 不进行下采样，第一个1x1逐点卷积层进行分组
    stride = 1
    downsample = nn.AvgPool2d(3, stride=stride, padding=1) if stride == 2 else None
    model = ShuffleNetV1Unit(planes, planes, groups, stride, downsample, with_group=True)
    print(model)

    outputs = model(outputs)
    print(outputs.shape)
    assert outputs.shape == (1, planes, 28, 28)


if __name__ == '__main__':
    test_shufflenet_v1_unit()

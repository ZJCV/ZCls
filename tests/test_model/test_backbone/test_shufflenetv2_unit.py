# -*- coding: utf-8 -*-

"""
@date: 2020/12/25 上午10:14
@file: test_shufflenetv1_unit.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn
from zcls.model.backbones.shufflenetv2_unit import ShuffleNetV2Unit


def test_shufflenetv2_unit():
    data = torch.randn(1, 24, 56, 56)
    inplanes = 24
    planes = 116

    # 下采样
    stride = 2
    branch_planes = planes // 2
    downsample = nn.Sequential(
        nn.Conv2d(inplanes, branch_planes, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(branch_planes),
        nn.Conv2d(branch_planes, branch_planes, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(branch_planes),
        nn.ReLU(inplace=True)
    )
    model = ShuffleNetV2Unit(inplanes, branch_planes, stride, downsample)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, planes, 28, 28)

    # 不进行下采样
    stride = 1
    branch_planes = planes // 2
    downsample = None
    model = ShuffleNetV2Unit(branch_planes, branch_planes, stride, downsample)
    print(model)

    outputs = model(outputs)
    print(outputs.shape)
    assert outputs.shape == (1, planes, 28, 28)


if __name__ == '__main__':
    test_shufflenetv2_unit()

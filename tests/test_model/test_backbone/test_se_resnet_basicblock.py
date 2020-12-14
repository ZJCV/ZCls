# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午3:00
@file: test_basicblock.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn

from zcls.model.backbones.se_resnet_basicblock import SEResNetBasicBlock


def test_se_resnet_basicblock():
    data = torch.randn(1, 64, 56, 56)
    inplanes = 64
    planes = 128
    expansion = SEResNetBasicBlock.expansion
    with_se = True
    reduction = 16

    # 不进行下采样
    stride = 1
    downsample = nn.Sequential(
        nn.Conv2d(inplanes, planes * expansion, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * expansion),
    )
    model = SEResNetBasicBlock(inplanes, planes, stride, downsample, with_se=with_se, reduction=reduction)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 128, 56, 56)

    # 进行下采样
    stride = 2
    downsample = nn.Sequential(
        nn.Conv2d(inplanes, planes * expansion, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * expansion),
    )
    model = SEResNetBasicBlock(inplanes, planes, stride, downsample, with_se=with_se, reduction=reduction)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 128, 28, 28)


if __name__ == '__main__':
    test_se_resnet_basicblock()

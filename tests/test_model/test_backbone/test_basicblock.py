# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午3:00
@file: test_basicblock.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn

from zcls.model.backbones.basicblock import BasicBlock


def test_basicblock():
    data = torch.randn(1, 64, 56, 56)
    in_planes = 64
    out_planes = 128
    expansion = BasicBlock.expansion

    # 不进行下采样
    stride = 1
    down_sample = nn.Sequential(
        nn.Conv2d(in_planes, out_planes * expansion, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_planes * expansion),
    )
    model = BasicBlock(in_planes=in_planes,
                       out_planes=out_planes,
                       stride=stride,
                       down_sample=down_sample)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 128, 56, 56)

    # 进行下采样
    stride = 2
    down_sample = nn.Sequential(
        nn.Conv2d(in_planes, out_planes * expansion, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_planes * expansion),
    )
    model = BasicBlock(in_planes=in_planes,
                       out_planes=out_planes,
                       stride=stride,
                       down_sample=down_sample)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 128, 28, 28)


if __name__ == '__main__':
    test_basicblock()

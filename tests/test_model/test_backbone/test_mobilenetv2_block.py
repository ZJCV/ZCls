# -*- coding: utf-8 -*-

"""
@date: 2020/12/4 下午3:34
@file: test_mobilenetv2_block.py
@author: zj
@description: 
"""

import torch

from zcls.model.backbones.mobilenet.mobilenetv2_block import MobileNetV2Block


def test_mobilenetv2_block():
    # 不进行下采样
    data = torch.randn(1, 32, 112, 112)
    in_planes = 32
    planes = 16
    stride = 1
    model = MobileNetV2Block(
        # 输入通道数
        in_planes,
        # 输出通道数
        planes,
        # 膨胀因子
        expansion_rate=1,
        # 重复次数
        repeat=1,
        # 卷积层步长
        stride=stride
    )
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 16, 112, 112)

    # 进行下采样
    data = torch.randn(1, 16, 112, 112)
    in_planes = 16
    planes = 24
    stride = 2
    model = MobileNetV2Block(
        # 输入通道数
        in_planes,
        # 输出通道数
        planes,
        # 膨胀因子
        expansion_rate=6,
        # 重复次数
        repeat=2,
        # 卷积层步长
        stride=stride
    )
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 24, 56, 56)


if __name__ == '__main__':
    test_mobilenetv2_block()

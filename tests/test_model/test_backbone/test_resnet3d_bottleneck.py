# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午3:33
@file: test_bottleneck.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn
from zcls.model.backbones.resnet3d_bottleneck import ResNet3DBottleneck


def test_resnet3d_bottleneck():
    data = torch.randn(1, 256, 1, 56, 56)
    inplanes = 256
    planes = 128
    expansion = ResNet3DBottleneck.expansion

    # 不膨胀
    # 不进行下采样
    temporal_stride = 1
    spatial_stride = 1
    downsample = nn.Sequential(
        nn.Conv3d(inplanes, planes * expansion, kernel_size=1,
                  stride=(temporal_stride, spatial_stride, spatial_stride), bias=False),
        nn.BatchNorm3d(planes * expansion),
    )
    model = ResNet3DBottleneck(inplanes=inplanes,
                               planes=planes,
                               spatial_stride=spatial_stride,
                               temporal_stride=temporal_stride,
                               inflate=False,
                               downsample=downsample)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, planes * expansion, 1, 56, 56)

    # 进行下采样
    temporal_stride = 1
    spatial_stride = 2
    downsample = nn.Sequential(
        nn.Conv3d(inplanes, planes * expansion, kernel_size=1,
                  stride=(temporal_stride, spatial_stride, spatial_stride), bias=False),
        nn.BatchNorm3d(planes * expansion),
    )
    model = ResNet3DBottleneck(inplanes=inplanes,
                               planes=planes,
                               spatial_stride=spatial_stride,
                               temporal_stride=temporal_stride,
                               inflate=False,
                               downsample=downsample)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, planes * expansion, 1, 28, 28)

    # 32x4d
    # 进行下采样
    temporal_stride = 1
    spatial_stride = 2
    downsample = nn.Sequential(
        nn.Conv3d(inplanes, planes * expansion, kernel_size=1,
                  stride=(temporal_stride, spatial_stride, spatial_stride), bias=False),
        nn.BatchNorm3d(planes * expansion),
    )
    model = ResNet3DBottleneck(inplanes=inplanes,
                               planes=planes,
                               spatial_stride=spatial_stride,
                               temporal_stride=temporal_stride,
                               inflate=False,
                               downsample=downsample,
                               groups=32,
                               base_width=4)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, planes * expansion, 1, 28, 28)


def test_resnet3d_bottleneck_3_1_1():
    data = torch.randn(1, 64, 8, 56, 56)
    inplanes = 64
    planes = 128
    expansion = ResNet3DBottleneck.expansion
    inflate_style = '3x1x1'

    # 膨胀，不进行时间下采样
    # 不进行下采样
    temporal_stride = 1
    spatial_stride = 1
    downsample = nn.Sequential(
        nn.Conv3d(inplanes, planes * expansion, kernel_size=1,
                  stride=(temporal_stride, spatial_stride, spatial_stride), bias=False),
        nn.BatchNorm3d(planes * expansion),
    )
    model = ResNet3DBottleneck(inplanes=inplanes,
                               planes=planes,
                               spatial_stride=spatial_stride,
                               temporal_stride=temporal_stride,
                               inflate=True,
                               inflate_style=inflate_style,
                               downsample=downsample)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, planes * expansion, 8, 56, 56)

    # 膨胀，进行时间下采样
    # 不进行下采样
    temporal_stride = 2
    spatial_stride = 1
    downsample = nn.Sequential(
        nn.Conv3d(inplanes, planes * expansion, kernel_size=1,
                  stride=(temporal_stride, spatial_stride, spatial_stride), bias=False),
        nn.BatchNorm3d(planes * expansion),
    )
    model = ResNet3DBottleneck(inplanes=inplanes,
                               planes=planes,
                               spatial_stride=spatial_stride,
                               temporal_stride=temporal_stride,
                               inflate=True,
                               inflate_style=inflate_style,
                               downsample=downsample)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, planes * expansion, 4, 56, 56)

    # 膨胀，进行时间下采样
    # 进行下采样
    temporal_stride = 2
    spatial_stride = 2
    downsample = nn.Sequential(
        nn.Conv3d(inplanes, planes * expansion, kernel_size=1,
                  stride=(temporal_stride, spatial_stride, spatial_stride), bias=False),
        nn.BatchNorm3d(planes * expansion),
    )
    model = ResNet3DBottleneck(inplanes=inplanes,
                               planes=planes,
                               spatial_stride=spatial_stride,
                               temporal_stride=temporal_stride,
                               inflate=True,
                               inflate_style=inflate_style,
                               downsample=downsample)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, planes * expansion, 4, 28, 28)


def test_resnet3d_bottleneck_3_3_3():
    data = torch.randn(1, 64, 8, 56, 56)
    inplanes = 64
    planes = 128
    expansion = ResNet3DBottleneck.expansion
    inflate_style = '3x3x3'

    # 膨胀，不进行时间下采样
    # 不进行下采样
    temporal_stride = 1
    spatial_stride = 1
    downsample = nn.Sequential(
        nn.Conv3d(inplanes, planes * expansion, kernel_size=1,
                  stride=(temporal_stride, spatial_stride, spatial_stride), bias=False),
        nn.BatchNorm3d(planes * expansion),
    )
    model = ResNet3DBottleneck(inplanes=inplanes,
                               planes=planes,
                               spatial_stride=spatial_stride,
                               temporal_stride=temporal_stride,
                               inflate=True,
                               inflate_style=inflate_style,
                               downsample=downsample)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, planes * expansion, 8, 56, 56)

    # 膨胀，进行时间下采样
    # 不进行下采样
    temporal_stride = 2
    spatial_stride = 1
    downsample = nn.Sequential(
        nn.Conv3d(inplanes, planes * expansion, kernel_size=1,
                  stride=(temporal_stride, spatial_stride, spatial_stride), bias=False),
        nn.BatchNorm3d(planes * expansion),
    )
    model = ResNet3DBottleneck(inplanes=inplanes,
                               planes=planes,
                               spatial_stride=spatial_stride,
                               temporal_stride=temporal_stride,
                               inflate=True,
                               inflate_style=inflate_style,
                               downsample=downsample)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, planes * expansion, 4, 56, 56)

    # 膨胀，进行时间下采样
    # 进行下采样
    temporal_stride = 2
    spatial_stride = 2
    downsample = nn.Sequential(
        nn.Conv3d(inplanes, planes * expansion, kernel_size=1,
                  stride=(temporal_stride, spatial_stride, spatial_stride), bias=False),
        nn.BatchNorm3d(planes * expansion),
    )
    model = ResNet3DBottleneck(inplanes=inplanes,
                               planes=planes,
                               spatial_stride=spatial_stride,
                               temporal_stride=temporal_stride,
                               inflate=True,
                               inflate_style=inflate_style,
                               downsample=downsample)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, planes * expansion, 4, 28, 28)


if __name__ == '__main__':
    print('*' * 100)
    test_resnet3d_bottleneck()
    print('*' * 100)
    test_resnet3d_bottleneck_3_1_1()
    print('*' * 100)
    test_resnet3d_bottleneck_3_3_3()

# -*- coding: utf-8 -*-

"""
@date: 2020/12/9 上午11:54
@file: test_resnet3d_basicblock.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn

from zcls.model.backbones.resnet3d_basicblock import ResNet3DBasicBlock


def test_resnet3d_basicblock():
    data = torch.randn(1, 64, 1, 56, 56)
    inplanes = 64
    planes = 128
    expansion = ResNet3DBasicBlock.expansion

    # 不膨胀
    # 不进行空间下采样
    temporal_stride = 1
    spatial_stride = 1
    downsample = nn.Sequential(
        nn.Conv3d(inplanes, planes * expansion, kernel_size=1,
                  stride=(temporal_stride, spatial_stride, spatial_stride), bias=False),
        nn.BatchNorm3d(planes * expansion),
    )
    model = ResNet3DBasicBlock(in_planes=inplanes,
                               out_planes=planes,
                               spatial_stride=spatial_stride,
                               temporal_stride=temporal_stride,
                               inflate=False,
                               down_sample=downsample)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, planes * expansion, 1, 56, 56)

    # 不膨胀
    # 进行空间下采样
    temporal_stride = 1
    spatial_stride = 2
    downsample = nn.Sequential(
        nn.Conv3d(inplanes, planes * expansion, kernel_size=1,
                  stride=(temporal_stride, spatial_stride, spatial_stride), bias=False),
        nn.BatchNorm3d(planes * expansion),
    )
    model = ResNet3DBasicBlock(in_planes=inplanes,
                               out_planes=planes,
                               spatial_stride=spatial_stride,
                               temporal_stride=temporal_stride,
                               inflate=False,
                               down_sample=downsample)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, planes * expansion, 1, 28, 28)


def test_resnet3d_basicblock_3_1_1():
    # 膨胀，不进行时间下采样
    # 不进行下采样
    data = torch.randn(1, 64, 8, 56, 56)
    inplanes = 64
    planes = 128
    inflate_style = '3x1x1'
    expansion = ResNet3DBasicBlock.expansion

    temporal_stride = 1
    spatial_stride = 1
    downsample = nn.Sequential(
        nn.Conv3d(inplanes, planes * expansion, kernel_size=1,
                  stride=(temporal_stride, spatial_stride, spatial_stride), bias=False),
        nn.BatchNorm3d(planes * expansion),
    )
    model = ResNet3DBasicBlock(in_planes=inplanes,
                               out_planes=planes,
                               spatial_stride=spatial_stride,
                               temporal_stride=temporal_stride,
                               inflate=True,
                               inflate_style=inflate_style,
                               down_sample=downsample)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, planes * expansion, 8, 56, 56)

    # 膨胀，进行时间下采样
    # 不进行下采样
    data = torch.randn(1, 64, 8, 56, 56)
    inplanes = 64
    planes = 128
    expansion = ResNet3DBasicBlock.expansion

    temporal_stride = 2
    spatial_stride = 1
    downsample = nn.Sequential(
        nn.Conv3d(inplanes, planes * expansion, kernel_size=1,
                  stride=(temporal_stride, spatial_stride, spatial_stride), bias=False),
        nn.BatchNorm3d(planes * expansion),
    )
    model = ResNet3DBasicBlock(in_planes=inplanes,
                               out_planes=planes,
                               spatial_stride=spatial_stride,
                               temporal_stride=temporal_stride,
                               inflate=True,
                               inflate_style=inflate_style,
                               down_sample=downsample)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, planes * expansion, 4, 56, 56)

    # 膨胀，进行时间下采样
    # 进行下采样
    data = torch.randn(1, 64, 8, 56, 56)
    inplanes = 64
    planes = 128
    expansion = ResNet3DBasicBlock.expansion

    temporal_stride = 2
    spatial_stride = 2
    downsample = nn.Sequential(
        nn.Conv3d(inplanes, planes * expansion, kernel_size=1,
                  stride=(temporal_stride, spatial_stride, spatial_stride), bias=False),
        nn.BatchNorm3d(planes * expansion),
    )
    model = ResNet3DBasicBlock(in_planes=inplanes,
                               out_planes=planes,
                               spatial_stride=spatial_stride,
                               temporal_stride=temporal_stride,
                               inflate=True,
                               inflate_style=inflate_style,
                               down_sample=downsample)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, planes * expansion, 4, 28, 28)


def test_resnet3d_basicblock_3_3_3():
    # 膨胀，不进行时间下采样
    # 不进行下采样
    data = torch.randn(1, 64, 8, 56, 56)
    inplanes = 64
    planes = 128
    inflate_style = '3x3x3'
    expansion = ResNet3DBasicBlock.expansion

    temporal_stride = 1
    spatial_stride = 1
    downsample = nn.Sequential(
        nn.Conv3d(inplanes, planes * expansion, kernel_size=1,
                  stride=(temporal_stride, spatial_stride, spatial_stride), bias=False),
        nn.BatchNorm3d(planes * expansion),
    )
    model = ResNet3DBasicBlock(in_planes=inplanes,
                               out_planes=planes,
                               spatial_stride=spatial_stride,
                               temporal_stride=temporal_stride,
                               inflate=True,
                               inflate_style=inflate_style,
                               down_sample=downsample)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, planes * expansion, 8, 56, 56)

    # 膨胀，进行时间下采样
    # 不进行下采样
    data = torch.randn(1, 64, 8, 56, 56)
    inplanes = 64
    planes = 128
    expansion = ResNet3DBasicBlock.expansion

    temporal_stride = 2
    spatial_stride = 1
    downsample = nn.Sequential(
        nn.Conv3d(inplanes, planes * expansion, kernel_size=1,
                  stride=(temporal_stride, spatial_stride, spatial_stride), bias=False),
        nn.BatchNorm3d(planes * expansion),
    )
    model = ResNet3DBasicBlock(in_planes=inplanes,
                               out_planes=planes,
                               spatial_stride=spatial_stride,
                               temporal_stride=temporal_stride,
                               inflate=True,
                               inflate_style=inflate_style,
                               down_sample=downsample)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, planes * expansion, 4, 56, 56)

    # 膨胀，进行时间下采样
    # 进行下采样
    data = torch.randn(1, 64, 8, 56, 56)
    inplanes = 64
    planes = 128
    expansion = ResNet3DBasicBlock.expansion

    temporal_stride = 2
    spatial_stride = 2
    downsample = nn.Sequential(
        nn.Conv3d(inplanes, planes * expansion, kernel_size=1,
                  stride=(temporal_stride, spatial_stride, spatial_stride), bias=False),
        nn.BatchNorm3d(planes * expansion),
    )
    model = ResNet3DBasicBlock(in_planes=inplanes,
                               out_planes=planes,
                               spatial_stride=spatial_stride,
                               temporal_stride=temporal_stride,
                               inflate=True,
                               inflate_style=inflate_style,
                               down_sample=downsample)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, planes * expansion, 4, 28, 28)


if __name__ == '__main__':
    print('*' * 100)
    test_resnet3d_basicblock()
    print('*' * 100)
    test_resnet3d_basicblock_3_1_1()
    print('*' * 100)
    test_resnet3d_basicblock_3_3_3()

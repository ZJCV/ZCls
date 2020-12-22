# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午3:00
@file: test_basicblock.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn

from zcls.model.backbones.attentation_resnet_bottleneck import AttentionResNetBottleneck


def test_se_resnet_basicblock():
    data = torch.randn(10, 64, 56, 56)
    inplanes = 64
    planes = 128
    expansion = AttentionResNetBottleneck.expansion
    with_attention = True
    reduction = 16
    attention_type = 'SqueezeAndExcitationBlock2D'

    # 不进行下采样
    stride = 1
    downsample = nn.Sequential(
        nn.Conv2d(inplanes, planes * expansion, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * expansion),
    )
    model = AttentionResNetBottleneck(inplanes, planes, stride, downsample, with_attention=with_attention,
                                      reduction=reduction, attention_type=attention_type)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (10, planes * expansion, 56, 56)

    # 进行下采样
    stride = 2
    downsample = nn.Sequential(
        nn.Conv2d(inplanes, planes * expansion, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * expansion),
    )
    model = AttentionResNetBottleneck(inplanes, planes, stride, downsample, with_attention=with_attention,
                                      reduction=reduction, attention_type=attention_type)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (10, planes * expansion, 28, 28)

    # 32x4d
    # 进行下采样
    stride = 2
    downsample = nn.Sequential(
        nn.Conv2d(inplanes, planes * expansion, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * expansion),
    )
    model = AttentionResNetBottleneck(inplanes, planes, stride, downsample, with_attention=with_attention,
                                      reduction=reduction, attention_type=attention_type,
                                      groups=32, base_width=4)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (10, planes * expansion, 28, 28)


def test_nl_resnet_basicblock():
    data = torch.randn(10, 64, 56, 56)
    inplanes = 64
    planes = 128
    expansion = AttentionResNetBottleneck.expansion
    with_attention = True
    reduction = 16
    attention_type = 'NonLocal2DEmbeddedGaussian'

    # 不进行下采样
    stride = 1
    downsample = nn.Sequential(
        nn.Conv2d(inplanes, planes * expansion, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * expansion),
    )
    model = AttentionResNetBottleneck(inplanes, planes, stride, downsample, with_attention=with_attention,
                                      reduction=reduction, attention_type=attention_type)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (10, planes * expansion, 56, 56)

    # 进行下采样
    stride = 2
    downsample = nn.Sequential(
        nn.Conv2d(inplanes, planes * expansion, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * expansion),
    )
    model = AttentionResNetBottleneck(inplanes, planes, stride, downsample, with_attention=with_attention,
                                      reduction=reduction, attention_type=attention_type)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (10, planes * expansion, 28, 28)

    # 32x4d
    # 进行下采样
    stride = 2
    downsample = nn.Sequential(
        nn.Conv2d(inplanes, planes * expansion, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * expansion),
    )
    model = AttentionResNetBottleneck(inplanes, planes, stride, downsample, with_attention=with_attention,
                                      reduction=reduction, attention_type=attention_type,
                                      groups=32, base_width=4)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (10, planes * expansion, 28, 28)


def test_snl_resnet_basicblock():
    data = torch.randn(10, 64, 56, 56)
    inplanes = 64
    planes = 128
    expansion = AttentionResNetBottleneck.expansion
    with_attention = True
    reduction = 16
    attention_type = 'SimplifiedNonLocal2DEmbeddedGaussian'

    # 不进行下采样
    stride = 1
    downsample = nn.Sequential(
        nn.Conv2d(inplanes, planes * expansion, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * expansion),
    )
    model = AttentionResNetBottleneck(inplanes, planes, stride, downsample, with_attention=with_attention,
                                      reduction=reduction, attention_type=attention_type)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (10, planes * expansion, 56, 56)

    # 进行下采样
    stride = 2
    downsample = nn.Sequential(
        nn.Conv2d(inplanes, planes * expansion, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * expansion),
    )
    model = AttentionResNetBottleneck(inplanes, planes, stride, downsample, with_attention=with_attention,
                                      reduction=reduction, attention_type=attention_type)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (10, planes * expansion, 28, 28)

    # 32x4d
    # 进行下采样
    stride = 2
    downsample = nn.Sequential(
        nn.Conv2d(inplanes, planes * expansion, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * expansion),
    )
    model = AttentionResNetBottleneck(inplanes, planes, stride, downsample, with_attention=with_attention,
                                      reduction=reduction, attention_type=attention_type,
                                      groups=32, base_width=4)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (10, planes * expansion, 28, 28)


def test_gc_resnet_basicblock():
    data = torch.randn(10, 64, 56, 56)
    inplanes = 64
    planes = 128
    expansion = AttentionResNetBottleneck.expansion
    with_attention = True
    reduction = 16
    attention_type = 'GlobalContextBlock2D'

    # 不进行下采样
    stride = 1
    downsample = nn.Sequential(
        nn.Conv2d(inplanes, planes * expansion, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * expansion),
    )
    model = AttentionResNetBottleneck(inplanes, planes, stride, downsample, with_attention=with_attention,
                                      reduction=reduction, attention_type=attention_type)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (10, planes * expansion, 56, 56)

    # 32x4d
    # 进行下采样
    stride = 2
    downsample = nn.Sequential(
        nn.Conv2d(inplanes, planes * expansion, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * expansion),
    )
    model = AttentionResNetBottleneck(inplanes, planes, stride, downsample, with_attention=with_attention,
                                      reduction=reduction, attention_type=attention_type,
                                      groups=32, base_width=4)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (10, planes * expansion, 28, 28)


if __name__ == '__main__':
    print('test_se_resnet_basicblock')
    test_se_resnet_basicblock()
    print('test_nl_resnet_basicblock')
    test_nl_resnet_basicblock()
    print('test_snl_resnet_basicblock')
    test_snl_resnet_basicblock()
    print('test_gc_resnet_basicblock')
    test_gc_resnet_basicblock()

# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午3:33
@file: test_bottleneck.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn
from zcls.model.backbones.resnet.bottleneck import Bottleneck


def test_bottleneck():
    data = torch.randn(1, 256, 56, 56)
    in_planes = 256
    out_planes = 128
    expansion = Bottleneck.expansion

    # 不进行下采样
    stride = 1
    down_sample = nn.Sequential(
        nn.Conv2d(in_planes, out_planes * expansion, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_planes * expansion),
    )
    model = Bottleneck(in_planes, out_planes, stride, down_sample)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 512, 56, 56)

    # 进行下采样
    stride = 2
    down_sample = nn.Sequential(
        nn.Conv2d(in_planes, out_planes * expansion, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_planes * expansion),
    )
    model = Bottleneck(in_planes, out_planes, stride, down_sample)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 512, 28, 28)

    # 32x4d
    # 进行下采样
    stride = 2
    down_sample = nn.Sequential(
        nn.Conv2d(in_planes, out_planes * expansion, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_planes * expansion),
    )
    model = Bottleneck(in_planes, out_planes, stride, down_sample, 32, 4)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 512, 28, 28)


def test_attention_bottleneck(attention_type='SqueezeAndExcitationBlock2D'):
    with_attention = 1
    reduction = 16

    data = torch.randn(3, 256, 56, 56)
    in_planes = 256
    out_planes = 128
    expansion = Bottleneck.expansion

    # 不进行下采样
    stride = 1
    down_sample = nn.Sequential(
        nn.Conv2d(in_planes, out_planes * expansion, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_planes * expansion),
    )
    model = Bottleneck(in_planes, out_planes, stride, down_sample)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (3, 512, 56, 56)

    # 进行下采样
    stride = 2
    down_sample = nn.Sequential(
        nn.Conv2d(in_planes, out_planes * expansion, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_planes * expansion),
    )
    model = Bottleneck(in_planes=in_planes,
                       out_planes=out_planes,
                       stride=stride,
                       down_sample=down_sample,
                       with_attention=with_attention,
                       reduction=reduction,
                       attention_type=attention_type
                       )
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (3, 512, 28, 28)

    # 32x4d
    # 进行下采样
    stride = 2
    down_sample = nn.Sequential(
        nn.Conv2d(in_planes, out_planes * expansion, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_planes * expansion),
    )
    model = Bottleneck(in_planes=in_planes,
                       out_planes=out_planes,
                       stride=stride,
                       down_sample=down_sample,
                       groups=32,
                       base_width=4,
                       with_attention=with_attention,
                       reduction=reduction,
                       attention_type=attention_type
                       )
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (3, 512, 28, 28)


def test_avg_bottleneck():
    data = torch.randn(1, 256, 56, 56)
    in_planes = 256
    out_planes = 128
    expansion = Bottleneck.expansion

    # 不进行下采样
    stride = 1
    down_sample = nn.Sequential(
        nn.Conv2d(in_planes, out_planes * expansion, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_planes * expansion),
    )
    model = Bottleneck(in_planes, out_planes, stride, down_sample, use_avg=False, fast_avg=False)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 512, 56, 56)

    # 进行下采样 avg
    stride = 2
    down_sample = nn.Sequential(
        nn.Conv2d(in_planes, out_planes * expansion, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_planes * expansion),
    )
    model = Bottleneck(in_planes, out_planes, stride, down_sample, use_avg=True, fast_avg=False)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 512, 28, 28)

    # 32x4d_fast_avg
    # 进行下采样
    stride = 2
    down_sample = nn.Sequential(
        nn.Conv2d(in_planes, out_planes * expansion, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_planes * expansion),
    )
    model = Bottleneck(in_planes, out_planes, stride, down_sample, 32, 4, use_avg=True, fast_avg=True)
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 512, 28, 28)


if __name__ == '__main__':
    print('*' * 10 + ' bottleneck')
    test_bottleneck()
    print('*' * 10 + ' se bottleneck')
    test_attention_bottleneck(attention_type='SqueezeAndExcitationBlock2D')
    print('*' * 10 + ' nl bottleneck')
    test_attention_bottleneck(attention_type='NonLocal2DEmbeddedGaussian')
    print('*' * 10 + ' snl bottleneck')
    test_attention_bottleneck(attention_type='SimplifiedNonLocal2DEmbeddedGaussian')
    print('*' * 10 + ' gc bottleneck')
    test_attention_bottleneck(attention_type='GlobalContextBlock2D')
    print('*' * 10 + ' avg bottleneck')
    test_avg_bottleneck()

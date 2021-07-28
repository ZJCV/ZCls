# -*- coding: utf-8 -*-

"""
@date: 2021/2/1 下午8:01
@file: test_asymmetric_convolution_block.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn

from zcls.config import cfg
from zcls.config.key_word import KEY_OUTPUT
from zcls.model.recognizers.vgg.repvgg import RepVGG
from zcls.model.layers.repvgg_block import RepVGGBlock
from zcls.model.conv_helper import insert_repvgg_block, fuse_repvgg_block


def test_repvgg_block():
    in_channels = 32
    out_channels = 64
    dilation = 1
    groups = 1

    # inputs == outputs
    kernel_size = 3
    stride = 1
    padding = 1
    acblock = RepVGGBlock(in_channels,
                          out_channels,
                          kernel_size,
                          stride=stride,
                          padding=padding,
                          dilation=dilation,
                          groups=groups)

    data = torch.randn(1, in_channels, 56, 56)
    outputs = acblock.forward(data)

    _, _, h, w = data.shape[:4]
    _, _, h2, w2 = outputs.shape[:4]
    assert h == h2 and w == w2

    # 下采样
    kernel_size = 3
    stride = 2
    padding = 1
    acblock = RepVGGBlock(in_channels,
                          out_channels,
                          kernel_size,
                          stride=stride,
                          padding=padding,
                          dilation=dilation,
                          groups=groups)

    data = torch.randn(1, in_channels, 56, 56)
    outputs = acblock.forward(data)

    _, _, h, w = data.shape[:4]
    _, _, h2, w2 = outputs.shape[:4]
    assert h / 2 == h2 and w / 2 == w2

    # 下采样 + 分组卷积
    kernel_size = 3
    stride = 2
    padding = 1
    groups = 8
    acblock = RepVGGBlock(in_channels,
                          out_channels,
                          kernel_size,
                          stride=stride,
                          padding=padding,
                          dilation=dilation,
                          groups=groups)

    data = torch.randn(1, in_channels, 56, 56)
    outputs = acblock.forward(data)

    _, _, h, w = data.shape[:4]
    _, _, h2, w2 = outputs.shape[:4]
    assert h / 2 == h2 and w / 2 == w2


def test_conv_helper():
    in_channels = 32
    out_channels = 64
    dilation = 1

    # 下采样 + 分组卷积
    kernel_size = 3
    stride = 2
    padding = 1
    groups = 8

    model = nn.Sequential(
        nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, dilation=dilation, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    model.eval()
    print(model)

    data = torch.randn(1, in_channels, 56, 56)
    insert_repvgg_block(model)
    model.eval()
    train_outputs = model(data)
    print(model)

    fuse_repvgg_block(model)
    model.eval()
    eval_outputs = model(data)
    print(model)

    print(torch.sqrt(torch.sum((train_outputs - eval_outputs) ** 2)))
    print(torch.allclose(train_outputs, eval_outputs, atol=1e-8))
    assert torch.allclose(train_outputs, eval_outputs, atol=1e-8)


def test_regvgg():
    cfg.merge_from_file('configs/benchmarks/repvgg/repvgg_a0_infer_zcls_imagenet_224.yaml')
    cfg.MODEL.RECOGNIZER.PRELOADED = ""
    model = RepVGG(cfg)
    model.eval()
    print(model)

    data = torch.randn(1, 3, 224, 224)
    insert_repvgg_block(model)
    model.eval()
    train_outputs = model(data)[KEY_OUTPUT]
    print(model)

    fuse_repvgg_block(model)
    model.eval()
    eval_outputs = model(data)[KEY_OUTPUT]
    print(model)

    print(torch.sqrt(torch.sum((train_outputs - eval_outputs) ** 2)))
    print(torch.allclose(train_outputs, eval_outputs, atol=1e-7))
    assert torch.allclose(train_outputs, eval_outputs, atol=1e-7)


if __name__ == '__main__':
    print('#' * 10)
    test_repvgg_block()
    print('@' * 10)
    test_conv_helper()
    print('*' * 10)
    test_regvgg()

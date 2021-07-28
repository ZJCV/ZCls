# -*- coding: utf-8 -*-

"""
@date: 2021/7/28 下午6:10
@file: test_dbblock.py
@author: zj
@description: 
"""

import torch

import torch.nn as nn
from torchvision.models import resnet50

from zcls.model.layers.diverse_branch_block import DiverseBranchBlock
from zcls.model.conv_helper import insert_dbblock, fuse_dbblock


def test_dbblock():
    in_channels = 32
    out_channels = 64
    dilation = 1
    groups = 1

    # inputs == outputs
    kernel_size = 3
    stride = 1
    padding = 1
    dbblock = DiverseBranchBlock(in_channels,
                                 out_channels,
                                 kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 dilation=dilation,
                                 groups=groups)
    print(dbblock)

    data = torch.randn(1, in_channels, 56, 56)
    outputs = dbblock.forward(data)

    _, _, h, w = data.shape[:4]
    _, _, h2, w2 = outputs.shape[:4]
    assert h == h2 and w == w2

    # 下采样
    kernel_size = 3
    stride = 2
    padding = 1
    dbblock = DiverseBranchBlock(in_channels,
                                 out_channels,
                                 kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 dilation=dilation,
                                 groups=groups)
    print(dbblock)

    data = torch.randn(1, in_channels, 56, 56)
    outputs = dbblock.forward(data)

    _, _, h, w = data.shape[:4]
    _, _, h2, w2 = outputs.shape[:4]
    assert h / 2 == h2 and w / 2 == w2

    # 下采样 + 分组卷积
    kernel_size = 3
    stride = 2
    padding = 1
    groups = 8
    dbblock = DiverseBranchBlock(in_channels,
                                 out_channels,
                                 kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 dilation=dilation,
                                 groups=groups)
    print(dbblock)

    data = torch.randn(1, in_channels, 56, 56)
    outputs = dbblock.forward(data)

    _, _, h, w = data.shape[:4]
    _, _, h2, w2 = outputs.shape[:4]
    assert h / 2 == h2 and w / 2 == w2

    # 下采样 + 深度卷积
    kernel_size = 3
    stride = 2
    padding = 1
    in_channels = 32
    out_channels = 32
    groups = 32
    dbblock = DiverseBranchBlock(in_channels,
                                 out_channels,
                                 kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 dilation=dilation,
                                 groups=groups)
    print(dbblock)

    data = torch.randn(1, in_channels, 56, 56)
    outputs = dbblock.forward(data)

    _, _, h, w = data.shape[:4]
    _, _, h2, w2 = outputs.shape[:4]
    assert h / 2 == h2 and w / 2 == w2


def test_dbb_helper():
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
            nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size),
                      stride=(stride, stride), padding=padding, dilation=(dilation, dilation), groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    print(model)

    data = torch.randn(1, in_channels, 56, 56)

    model.train()
    insert_dbblock(model)
    print(model)

    model.eval()
    train_outputs = model(data)

    model.train()
    fuse_dbblock(model)
    model.eval()
    eval_outputs = model(data)
    print(model)

    print(torch.sqrt(torch.sum((train_outputs - eval_outputs) ** 2)))
    print(torch.allclose(train_outputs, eval_outputs, atol=1e-6))
    assert torch.allclose(train_outputs, eval_outputs, atol=1e-6)


def test_resnet50_dbb():
    model = resnet50()
    print('src model: ' + '*' * 10)
    print(model)

    data = torch.randn(1, 3, 224, 224)
    insert_dbblock(model)
    model.eval()
    train_outputs = model(data)
    print('inserted model: ' + '*' * 10)
    print(model)

    model.train()
    fuse_dbblock(model)
    model.eval()
    eval_outputs = model(data)
    print('fused model: ' + '*' * 10)
    print(model)

    print(torch.sqrt(torch.sum((train_outputs - eval_outputs) ** 2)))
    print(torch.allclose(train_outputs, eval_outputs, atol=1e-4))
    assert torch.allclose(train_outputs, eval_outputs, atol=1e-4)


if __name__ == '__main__':
    # test_dbblock()
    # test_dbb_helper()
    test_resnet50_dbb()

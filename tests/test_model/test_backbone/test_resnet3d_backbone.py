# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午7:24
@file: test_resnet_backbone.py
@author: zj
@description: 
"""

import torch
from zcls.model.backbones.resnet3d_basicblock import ResNet3DBasicBlock
from zcls.model.backbones.resnet3d_bottleneck import ResNet3DBottleneck
from zcls.model.backbones.resnet3d_backbone import ResNet3DBackbone


def test_resnet3d_backbone():
    # 不执行膨胀操作
    # for R18
    model = ResNet3DBackbone(
        layer_blocks=(2, 2, 2, 2),
        block_layer=ResNet3DBasicBlock,
        zero_init_residual=True
    )
    print(model)

    data = torch.randn(1, 3, 1, 224, 224)
    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 512, 1, 7, 7)

    # for R50
    model = ResNet3DBackbone(
        layer_blocks=(3, 4, 6, 3),
        block_layer=ResNet3DBottleneck,
        zero_init_residual=True
    )
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 2048, 1, 7, 7)


def test_pretrained_resnet3d_backbone():
    from torchvision.models.utils import load_state_dict_from_url
    state_dict_2d = load_state_dict_from_url('https://download.pytorch.org/models/resnet50-19c8e357.pth', progress=True)

    data = torch.randn(1, 3, 1, 224, 224)
    # for R50
    model = ResNet3DBackbone(
        layer_blocks=(3, 4, 6, 3),
        block_layer=ResNet3DBottleneck,
        zero_init_residual=True,
        state_dict_2d=state_dict_2d
    )
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 2048, 1, 7, 7)


def test_resnet3d_backbone_c2d():
    from torchvision.models.utils import load_state_dict_from_url
    state_dict_2d = load_state_dict_from_url('https://download.pytorch.org/models/resnet50-19c8e357.pth', progress=True)

    data = torch.randn(1, 3, 32, 224, 224)
    # for R50
    model = ResNet3DBackbone(
        inplanes=3,
        base_planes=64,
        conv1_kernel=(1, 7, 7),
        conv1_stride=(2, 2, 2),
        conv1_padding=(0, 3, 3),
        pool1_kernel=(3, 3, 3),
        pool1_stride=(2, 2, 2),
        pool1_padding=(0, 1, 1),
        with_pool2=True,
        layer_planes=(64, 128, 256, 512),
        layer_blocks=(3, 4, 6, 3),
        downsamples=(0, 1, 1, 1),
        temporal_strides=(1, 1, 1, 1),
        inflate_list=(0, 0, 0, 0),
        inflate_style='3x1x1',
        block_layer=ResNet3DBottleneck,
        zero_init_residual=True,
        state_dict_2d=state_dict_2d
    )
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 2048, 4, 7, 7)


def test_resnet3d_backbone_i3d_3x1x1():
    from torchvision.models.utils import load_state_dict_from_url
    state_dict_2d = load_state_dict_from_url('https://download.pytorch.org/models/resnet50-19c8e357.pth', progress=True)

    data = torch.randn(1, 3, 32, 224, 224)
    # for R50
    model = ResNet3DBackbone(
        inplanes=3,
        base_planes=64,
        conv1_kernel=(5, 7, 7),
        conv1_stride=(2, 2, 2),
        conv1_padding=(2, 3, 3),
        pool1_kernel=(3, 3, 3),
        pool1_stride=(2, 2, 2),
        pool1_padding=(0, 1, 1),
        with_pool2=True,
        layer_planes=(64, 128, 256, 512),
        layer_blocks=(3, 4, 6, 3),
        downsamples=(0, 1, 1, 1),
        temporal_strides=(1, 1, 1, 1),
        inflate_list=((1, 0, 1), (0, 1, 0, 1), (0, 1, 0, 1, 0, 1), (0, 1, 0)),
        inflate_style='3x1x1',
        block_layer=ResNet3DBottleneck,
        zero_init_residual=True,
        state_dict_2d=state_dict_2d
    )
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 2048, 4, 7, 7)


def test_resnet3d_backbone_i3d_3x3x3():
    from torchvision.models.utils import load_state_dict_from_url
    state_dict_2d = load_state_dict_from_url('https://download.pytorch.org/models/resnet50-19c8e357.pth', progress=True)

    data = torch.randn(1, 3, 32, 224, 224)
    # for R50
    model = ResNet3DBackbone(
        inplanes=3,
        base_planes=64,
        conv1_kernel=(5, 7, 7),
        conv1_stride=(2, 2, 2),
        conv1_padding=(2, 3, 3),
        pool1_kernel=(3, 3, 3),
        pool1_stride=(2, 2, 2),
        pool1_padding=(0, 1, 1),
        with_pool2=True,
        layer_planes=(64, 128, 256, 512),
        layer_blocks=(3, 4, 6, 3),
        downsamples=(0, 1, 1, 1),
        temporal_strides=(1, 1, 1, 1),
        inflate_list=((1, 0, 1), (0, 1, 0, 1), (0, 1, 0, 1, 0, 1), (0, 1, 0)),
        inflate_style='3x3x3',
        block_layer=ResNet3DBottleneck,
        zero_init_residual=True,
        state_dict_2d=state_dict_2d
    )
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 2048, 4, 7, 7)


if __name__ == '__main__':
    test_resnet3d_backbone()
    print('*' * 100)
    test_pretrained_resnet3d_backbone()
    print('*' * 100)
    test_resnet3d_backbone_c2d()
    print('*' * 100)
    test_resnet3d_backbone_i3d_3x1x1()
    print('*' * 100)
    test_resnet3d_backbone_i3d_3x3x3()

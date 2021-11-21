# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午4:16
@file: test_resnest.py
@author: zj
@description: 
"""

import torch

from zcls.config import cfg
from zcls.config.key_word import KEY_OUTPUT
from zcls.model.recognizers.resnet.resnet import ResNet
from zcls.model.recognizers.resnet.torchvision_resnet import build_torchvision_resnet


def test_data(model, input_shape, output_shape):
    data = torch.randn(input_shape)
    outputs = model(data)[KEY_OUTPUT]
    print(outputs.shape)

    assert outputs.shape == output_shape


def test_resnet():
    config_file = 'configs/benchmarks/resnet-resnext/r18_zcls_imagenet_224.yaml'
    cfg.merge_from_file(config_file)

    model = ResNet(cfg)
    print(model)
    test_data(model, (1, 3, 224, 224), (1, 1000))

    config_file = 'configs/benchmarks/resnet-resnext/r18_torchvision_imagenet_224.yaml'
    cfg.merge_from_file(config_file)

    model = build_torchvision_resnet(cfg)
    print(model)
    test_data(model, (1, 3, 224, 224), (1, 1000))

    config_file = 'configs/benchmarks/resnet-resnext/rxt50_32x4d_zcls_imagenet_224.yaml'
    cfg.merge_from_file(config_file)

    model = ResNet(cfg)
    print(model)
    test_data(model, (1, 3, 224, 224), (1, 1000))

    config_file = 'configs/benchmarks/resnet-resnext/rxt50_32x4d_torchvision_imagenet_224.yaml'
    cfg.merge_from_file(config_file)

    model = build_torchvision_resnet(cfg)
    print(model)
    test_data(model, (1, 3, 224, 224), (1, 1000))


if __name__ == '__main__':
    print('*' * 10 + ' resnet-resnext')
    test_resnet()

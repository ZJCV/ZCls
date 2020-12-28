# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午4:16
@file: test_resnet.py
@author: zj
@description: 
"""

import torch

from zcls.config import cfg
from zcls.config.key_word import KEY_OUTPUT
from zcls.model.norm_helper import get_norm
from zcls.model.recognizers.resnet_recognizer import TorchvisionResNet, ResNetRecognizer


def test_data(model, input_shape, output_shape):
    data = torch.randn(input_shape)
    outputs = model(data)[KEY_OUTPUT]
    print(outputs.shape)

    assert outputs.shape == output_shape


def test_resnet():
    # for torchvision
    model = TorchvisionResNet(
        arch='resnet50',
        num_classes=1000
    )
    print(model)
    test_data(model, (1, 3, 224, 224), (1, 1000))

    # for custom
    model = ResNetRecognizer(
        arch="resnet50",
        num_classes=1000
    )
    print(model)
    test_data(model, (1, 3, 224, 224), (1, 1000))


def test_resnet_gn():
    cfg.MODEL.NORM.TYPE = 'GroupNorm'
    norm_layer = get_norm(cfg)
    print(norm_layer)

    # for custom
    model = ResNetRecognizer(
        arch="resnet50",
        num_classes=1000,
        norm_layer=norm_layer
    )
    print(model)
    test_data(model, (1, 3, 224, 224), (1, 1000))


if __name__ == '__main__':
    test_resnet()
    test_resnet_gn()

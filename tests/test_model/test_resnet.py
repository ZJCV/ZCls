# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午4:16
@file: test_resnet.py
@author: zj
@description: 
"""

import torch
from torchvision.models import resnet50

from zcls.config import cfg
from zcls.model.batchnorm_helper import get_norm
from zcls.model.recognizers.resnet_recognizer import ResNetRecognizer


def test_resnet():
    model = ResNetRecognizer(
        arch="resnet50",
        feature_dims=2048,
        num_classes=1000
    )
    print(model)

    data = torch.randn(1, 3, 224, 224)
    outputs = model(data)['probs']
    print(outputs.shape)

    assert outputs.shape == (1, 1000)

    model = resnet50()
    print(model)


def test_resnet_gn():
    cfg.MODEL.NORM.TYPE = 'GroupNorm'
    norm_layer = get_norm(cfg)
    print(norm_layer)

    model = ResNetRecognizer(
        arch="resnet50",
        feature_dims=2048,
        num_classes=1000,
        norm_layer=norm_layer
    )
    print(model)


if __name__ == '__main__':
    # test_resnet()
    test_resnet_gn()

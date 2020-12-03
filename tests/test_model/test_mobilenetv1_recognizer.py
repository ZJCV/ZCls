# -*- coding: utf-8 -*-

"""
@date: 2020/12/3 下午8:27
@file: test_mobilenetv1_recognizer.py
@author: zj
@description: 
"""

import torch

from zcls.config import cfg
from zcls.model.norm_helper import get_norm
from zcls.model.recognizers.mobilenetv1_recognizer import MobileNetV1Recognizer


def test_mobilenetv1():
    model = MobileNetV1Recognizer(
        feature_dims=1024,
        num_classes=1000
    )
    print(model)

    data = torch.randn(1, 3, 224, 224)
    outputs = model(data)['probs']
    print(outputs.shape)

    assert outputs.shape == (1, 1000)


def test_mobilenetv1_gn():
    cfg.MODEL.NORM.TYPE = 'GroupNorm'
    norm_layer = get_norm(cfg)
    print(norm_layer)

    model = MobileNetV1Recognizer(
        feature_dims=1024,
        num_classes=1000,
        norm_layer=norm_layer
    )
    print(model)


if __name__ == '__main__':
    test_mobilenetv1()
    test_mobilenetv1_gn()

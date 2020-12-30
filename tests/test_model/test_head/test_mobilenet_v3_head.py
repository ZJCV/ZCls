# -*- coding: utf-8 -*-

"""
@date: 2020/12/30 下午9:42
@file: test_mobilenet_v3_head.py
@author: zj
@description: 
"""

import torch

from zcls.model.heads.mobilenetv3_head import MobileNetV3Head


def test_mobilenet_v3_head():
    data = torch.randn(1, 960, 7, 7)

    model = MobileNetV3Head(
        feature_dims=960,
        inner_dims=1280,
        num_classes=1000,
        conv_layer=None,
        act_layer=None
    )
    print(model)

    outputs = model(data)
    print(outputs.shape)

    assert outputs.shape == (1, 1000)


if __name__ == '__main__':
    test_mobilenet_v3_head()

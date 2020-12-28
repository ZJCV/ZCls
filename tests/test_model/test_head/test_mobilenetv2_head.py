# -*- coding: utf-8 -*-

"""
@date: 2020/12/3 下午8:19
@file: test_mobilenetv1_head.py
@author: zj
@description: 
"""

import torch
from zcls.model.heads.mobilenetv2_head import MobileNetV2Head


def test_mobilenet_v2_head():
    data = torch.randn(1, 1280, 7, 7)

    feature_dims = 1280
    num_classes = 1000
    model = MobileNetV2Head(feature_dims=feature_dims, num_classes=num_classes)

    outputs = model(data)
    print(outputs.shape)

    assert outputs.shape == (1, 1000)


if __name__ == '__main__':
    test_mobilenet_v2_head()

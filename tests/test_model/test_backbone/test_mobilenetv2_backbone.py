# -*- coding: utf-8 -*-

"""
@date: 2020/12/3 下午7:42
@file: test_mobilenetv1_block.py
@author: zj
@description: 
"""

import torch

from zcls.model.backbones.mobilenet.mobilenetv2_backbone import MobileNetV2Backbone


def test_mobilenet_v2_backbone():
    data = torch.randn(1, 3, 224, 224)

    model = MobileNetV2Backbone()
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 1280, 7, 7)


if __name__ == '__main__':
    test_mobilenet_v2_backbone()

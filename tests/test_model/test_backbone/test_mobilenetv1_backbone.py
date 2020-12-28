# -*- coding: utf-8 -*-

"""
@date: 2020/12/3 下午7:42
@file: test_mobilenetv1_block.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn

from zcls.model.backbones.mobilenetv1_backbone import MobileNetV1Backbone


def test_mobilenet_v1_backbone():
    data = torch.randn(1, 3, 224, 224)

    model = MobileNetV1Backbone()
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 1024, 7, 7)


if __name__ == '__main__':
    test_mobilenet_v1_backbone()

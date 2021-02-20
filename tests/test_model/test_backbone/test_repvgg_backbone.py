# -*- coding: utf-8 -*-

"""
@date: 2021/2/2 下午5:02
@file: test_repvgg_backbone.py
@author: zj
@description: 
"""

import torch
from zcls.model.backbones.vgg.repvgg_backbone import RepVGGBackbone


def test_repvgg_backbone():
    model = RepVGGBackbone()
    print(model)

    data = torch.randn(1, 3, 224, 224)
    outputs = model(data)

    print(outputs.shape)
    assert outputs.shape == (1, 512, 7, 7)


if __name__ == '__main__':
    test_repvgg_backbone()

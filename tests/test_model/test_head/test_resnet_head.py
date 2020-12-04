# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午7:24
@file: test_resnet_head.py
@author: zj
@description: 
"""

import torch
from zcls.model.heads.resnet_head import ResNetHead


def test_resnet_head():
    data = torch.randn(1, 2048, 7, 7)

    feature_dims = 2048
    num_classes = 1000
    model = ResNetHead(feature_dims=feature_dims, num_classes=num_classes)

    outputs = model(data)
    print(outputs.shape)

    assert outputs.shape == (1, 1000)


if __name__ == '__main__':
    test_resnet_head()

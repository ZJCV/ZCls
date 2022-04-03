# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午11:14
@file: resnet.py
@author: zj
@description: 
"""

import torch.nn as nn
import torchvision.models as models

from .util import create_linear

__dict__ = [
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'
]


def get_resnet(pretrained=False, num_classes=1000, arch='resnet18'):
    assert arch in __dict__, f"{arch} not in {__dict__}"

    if pretrained == True:
        model = models.__dict__[arch](pretrained=True)
        if num_classes != 1000:
            old_fc = model.fc
            assert isinstance(old_fc, nn.Linear)

            in_features = old_fc.in_features
            new_fc = create_linear(in_features, num_classes, bias=old_fc.bias is not None)

            model.fc = new_fc
    else:
        model = models.__dict__[arch](pretrained=False, num_classes=num_classes)

    return model

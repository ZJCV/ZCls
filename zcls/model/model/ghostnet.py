# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午5:20
@file: ghostnet.py
@author: zj
@description: 
"""

import torch.nn as nn
import timm.models as models

from .util import create_linear


def get_ghostnet(pretrained=False, num_classes=1000, arch='ghostnet_050'):
    assert arch in ['ghostnet_050', 'ghostnet_100', 'ghostnet_130']

    if pretrained == True:
        model = models.__dict__[arch](pretrained=True, num_classes=1000)
        if num_classes != 1000:
            old_fc = model.classifier
            assert isinstance(old_fc, nn.Linear)

            in_features = old_fc.in_features
            new_fc = create_linear(in_features, num_classes, bias=old_fc.bias is not None)

            model.classifier = new_fc
    else:
        model = models.__dict__[arch](pretrained=False, num_classes=num_classes)

    return model


if __name__ == '__main__':
    model = get_ghostnet(num_classes=501, arch='ghostnet_130', pretrained=True)
    print(model)

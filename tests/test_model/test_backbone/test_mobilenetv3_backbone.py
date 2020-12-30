# -*- coding: utf-8 -*-

"""
@date: 2020/12/30 下午9:36
@file: test_mobilenetv3_backbone.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn

from zcls.model.backbones.mobilenetv3_backbone import MobileNetV3Backbone


def test_mobilenet_v3_backbone():
    data = torch.randn(1, 3, 224, 224)
    model = MobileNetV3Backbone(
        in_planes=3,
        base_planes=16,
        out_planes=960,
        width_multiplier=1.,
        round_nearest=8,
        reduction=4,
        attention_type='SqueezeAndExcitationBlock2D',
        conv_layer=None,
        norm_layer=None,
        act_layer=None,
    )
    print(model)
    outputs = model(data)
    print(outputs.shape)

    assert outputs.shape == (1, 960, 7, 7)


if __name__ == '__main__':
    test_mobilenet_v3_backbone()

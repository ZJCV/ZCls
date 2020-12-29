# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午7:24
@file: test_resnet_backbone.py
@author: zj
@description: 
"""

import torch
from zcls.model.backbones.shufflenetv2_unit import ShuffleNetV2Unit
from zcls.model.backbones.shufflenetv2_backbone import ShuffleNetV2Backbone


def test_shufflenet_v2_backbone():
    # 1x
    model = ShuffleNetV2Backbone(
        out_planes=1024,
        layer_planes=(116, 232, 464),
        layer_blocks=(4, 8, 4),
        down_samples=(1, 1, 1),
        block_layer=ShuffleNetV2Unit,
    )
    print(model)

    data = torch.randn(1, 3, 224, 224)
    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 1024, 7, 7)

    # 0.5x
    model = ShuffleNetV2Backbone(
        out_planes=1024,
        layer_planes=(48, 96, 192),
        layer_blocks=(4, 8, 4),
        down_samples=(1, 1, 1),
        block_layer=ShuffleNetV2Unit,
    )
    print(model)

    data = torch.randn(1, 3, 224, 224)
    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 1024, 7, 7)

    # 1.5x
    model = ShuffleNetV2Backbone(
        out_planes=1024,
        layer_planes=(176, 352, 704),
        layer_blocks=(4, 8, 4),
        down_samples=(1, 1, 1),
        block_layer=ShuffleNetV2Unit,
    )
    print(model)

    data = torch.randn(1, 3, 224, 224)
    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 1024, 7, 7)

    # 2x
    model = ShuffleNetV2Backbone(
        out_planes=2048,
        layer_planes=(244, 488, 976),
        layer_blocks=(4, 8, 4),
        down_samples=(1, 1, 1),
        block_layer=ShuffleNetV2Unit,
    )
    print(model)

    data = torch.randn(1, 3, 224, 224)
    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 2048, 7, 7)


if __name__ == '__main__':
    test_shufflenet_v2_backbone()

# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午7:24
@file: test_resnet_backbone.py
@author: zj
@description: 
"""

import torch
from zcls.model.backbones.shufflenet.shufflenetv2_unit import ShuffleNetV2Unit
from zcls.model.backbones.shufflenet.shufflenetv2_backbone import ShuffleNetV2Backbone


def test_shufflenet_v2_backbone():
    # 1x
    model = ShuffleNetV2Backbone(
        out_channels=1024,
        stage_channels=(116, 232, 464),
        stage_blocks=(4, 8, 4),
        downsamples=(1, 1, 1),
        block_layer=ShuffleNetV2Unit,
    )
    print(model)

    data = torch.randn(1, 3, 224, 224)
    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 1024, 7, 7)

    # 0.5x
    model = ShuffleNetV2Backbone(
        out_channels=1024,
        stage_channels=(48, 96, 192),
        stage_blocks=(4, 8, 4),
        downsamples=(1, 1, 1),
        block_layer=ShuffleNetV2Unit,
    )
    print(model)

    data = torch.randn(1, 3, 224, 224)
    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 1024, 7, 7)

    # 1.5x
    model = ShuffleNetV2Backbone(
        out_channels=1024,
        stage_channels=(176, 352, 704),
        stage_blocks=(4, 8, 4),
        downsamples=(1, 1, 1),
        block_layer=ShuffleNetV2Unit,
    )
    print(model)

    data = torch.randn(1, 3, 224, 224)
    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 1024, 7, 7)

    # 2x
    model = ShuffleNetV2Backbone(
        out_channels=2048,
        stage_channels=(244, 488, 976),
        stage_blocks=(4, 8, 4),
        downsamples=(1, 1, 1),
        block_layer=ShuffleNetV2Unit,
    )
    print(model)

    data = torch.randn(1, 3, 224, 224)
    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 2048, 7, 7)


if __name__ == '__main__':
    test_shufflenet_v2_backbone()

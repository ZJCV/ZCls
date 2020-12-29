# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午7:24
@file: test_resnet_backbone.py
@author: zj
@description: 
"""

import torch
from zcls.model.backbones.shufflenetv1_unit import ShuffleNetV1Unit
from zcls.model.backbones.shufflenetv1_backbone import ShuffleNetV1Backbone


def test_shufflenet_v1_backbone():
    # g=1
    model = ShuffleNetV1Backbone(
        groups=1,
        layer_planes=(144, 288, 576),
        layer_blocks=(4, 8, 4),
        down_samples=(1, 1, 1),
        with_groups=(0, 1, 1),
        block_layer=ShuffleNetV1Unit,
    )
    print(model)

    data = torch.randn(1, 3, 224, 224)
    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 576, 7, 7)

    # g=2
    model = ShuffleNetV1Backbone(
        groups=2,
        layer_planes=(200, 400, 800),
        layer_blocks=(4, 8, 4),
        down_samples=(1, 1, 1),
        with_groups=(0, 1, 1),
        block_layer=ShuffleNetV1Unit,
    )
    print(model)

    data = torch.randn(1, 3, 224, 224)
    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 800, 7, 7)

    # g=3
    model = ShuffleNetV1Backbone(
        groups=3,
        layer_planes=(240, 480, 960),
        layer_blocks=(4, 8, 4),
        down_samples=(1, 1, 1),
        with_groups=(0, 1, 1),
        block_layer=ShuffleNetV1Unit,
    )
    print(model)

    data = torch.randn(1, 3, 224, 224)
    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 960, 7, 7)

    # g=4
    model = ShuffleNetV1Backbone(
        groups=4,
        layer_planes=(272, 544, 1088),
        layer_blocks=(4, 8, 4),
        down_samples=(1, 1, 1),
        with_groups=(0, 1, 1),
        block_layer=ShuffleNetV1Unit,
    )
    print(model)

    data = torch.randn(1, 3, 224, 224)
    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 1088, 7, 7)

    # g=8
    model = ShuffleNetV1Backbone(
        groups=8,
        layer_planes=(384, 768, 1536),
        layer_blocks=(4, 8, 4),
        down_samples=(1, 1, 1),
        with_groups=(0, 1, 1),
        block_layer=ShuffleNetV1Unit,
    )
    print(model)

    data = torch.randn(1, 3, 224, 224)
    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 1536, 7, 7)


if __name__ == '__main__':
    test_shufflenet_v1_backbone()

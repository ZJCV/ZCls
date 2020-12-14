# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午7:24
@file: test_resnet_backbone.py
@author: zj
@description: 
"""

import torch
from zcls.model.backbones.se_resnet_basicblock import SEResNetBasicBlock
from zcls.model.backbones.se_resnet_bottleneck import SEResNetBottleneck
from zcls.model.backbones.se_resnet_backbone import SEResNetBackbone


def test_se_resnet_backbone():
    # for R18
    model = SEResNetBackbone(
        layer_blocks=(2, 2, 2, 2),
        block_layer=SEResNetBasicBlock,
    )
    print(model)

    data = torch.randn(1, 3, 224, 224)
    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 512, 7, 7)

    # for R50
    model = SEResNetBackbone(
        layer_blocks=(3, 4, 6, 3),
        block_layer=SEResNetBottleneck,
    )
    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 2048, 7, 7)


if __name__ == '__main__':
    test_se_resnet_backbone()

# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午7:24
@file: test_resnet_backbone.py
@author: zj
@description: 
"""

import torch
from zcls.model.backbones.attentation_resnet_basicblock import AttentionResNetBasicBlock
from zcls.model.backbones.attentation_resnet_bottleneck import AttentionResNetBottleneck
from zcls.model.backbones.attention_resnet_backbone import AttentionResNetBackbone


def test_attention_resnet_backbone():
    # for R18
    # se
    model = AttentionResNetBackbone(
        layer_blocks=(2, 2, 2, 2),
        with_attention=(1, 1, 1, 1),
        reduction=16,
        attention_type='SqueezeAndExcitationBlock2D',
        block_layer=AttentionResNetBasicBlock,
    )
    print(model)

    data = torch.randn(10, 3, 224, 224)
    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (10, 512, 7, 7)

    # nl
    model = AttentionResNetBackbone(
        layer_blocks=(2, 2, 2, 2),
        with_attention=((0, 0), (1, 1), (1, 1), (0, 0)),
        attention_type='NonLocal2DEmbeddedGaussian',
        block_layer=AttentionResNetBasicBlock,
    )
    print(model)

    data = torch.randn(10, 3, 224, 224)
    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (10, 512, 7, 7)

    # snl
    model = AttentionResNetBackbone(
        layer_blocks=(2, 2, 2, 2),
        with_attention=((0, 0), (1, 1), (1, 1), (0, 0)),
        attention_type='SimplifiedNonLocal2DEmbeddedGaussian',
        block_layer=AttentionResNetBasicBlock,
    )
    print(model)

    data = torch.randn(10, 3, 224, 224)
    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (10, 512, 7, 7)

    # gc
    model = AttentionResNetBackbone(
        layer_blocks=(2, 2, 2, 2),
        with_attention=(1, 1, 1, 1),
        reduction=16,
        attention_type='GlobalContextBlock2D',
        block_layer=AttentionResNetBasicBlock,
    )
    print(model)

    data = torch.randn(10, 3, 224, 224)
    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (10, 512, 7, 7)

    # for R50
    # se
    model = AttentionResNetBackbone(
        layer_blocks=(3, 4, 6, 3),
        with_attention=(1, 1, 1, 1),
        reduction=16,
        attention_type='SqueezeAndExcitationBlock2D',
        block_layer=AttentionResNetBottleneck,
    )
    print(model)

    data = torch.randn(10, 3, 224, 224)
    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (10, 2048, 7, 7)

    # nl
    model = AttentionResNetBackbone(
        layer_blocks=(3, 4, 6, 3),
        with_attention=(0, (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), 0),
        attention_type='NonLocal2DEmbeddedGaussian',
        block_layer=AttentionResNetBottleneck,
    )
    print(model)

    data = torch.randn(10, 3, 224, 224)
    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (10, 2048, 7, 7)

    # snl
    model = AttentionResNetBackbone(
        layer_blocks=(3, 4, 6, 3),
        with_attention=(0, (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), 0),
        attention_type='SimplifiedNonLocal2DEmbeddedGaussian',
        block_layer=AttentionResNetBottleneck,
    )
    print(model)

    data = torch.randn(10, 3, 224, 224)
    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (10, 2048, 7, 7)

    # gc
    model = AttentionResNetBackbone(
        layer_blocks=(3, 4, 6, 3),
        with_attention=(1, 1, 1, 1),
        reduction=16,
        attention_type='GlobalContextBlock2D',
        block_layer=AttentionResNetBottleneck,
    )
    print(model)

    data = torch.randn(10, 3, 224, 224)
    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (10, 2048, 7, 7)


if __name__ == '__main__':
    test_attention_resnet_backbone()

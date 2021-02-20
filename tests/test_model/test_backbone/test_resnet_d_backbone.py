# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午7:24
@file: test_resnet_backbone.py
@author: zj
@description: 
"""

import torch
from zcls.model.backbones.resnet.basicblock import BasicBlock
from zcls.model.backbones.resnet.bottleneck import Bottleneck
from zcls.model.backbones.resnet.resnet_d_backbone import ResNetDBackbone


def test_resnet_d_backbone():
    # for R18
    model = ResNetDBackbone(
        layer_blocks=(2, 2, 2, 2),
        block_layer=BasicBlock,
    )
    print(model)

    data = torch.randn(1, 3, 224, 224)
    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 512, 7, 7)

    # for R50
    model = ResNetDBackbone(
        layer_blocks=(3, 4, 6, 3),
        block_layer=Bottleneck,
    )
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 2048, 7, 7)

    # for RX50_32x4d
    model = ResNetDBackbone(
        layer_blocks=(3, 4, 6, 3),
        groups=32,
        width_per_group=4,
        block_layer=Bottleneck,
    )
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 2048, 7, 7)


def test_attention_resnet_d_backbone(with_attentions=(1, 1, 1, 1),
                                     reduction=16,
                                     attention_type='SqueezeAndExcitationBlock2D'
                                     ):
    data = torch.randn(3, 3, 224, 224)

    # for R50
    model = ResNetDBackbone(
        layer_blocks=(3, 4, 6, 3),
        block_layer=Bottleneck,
        with_attentions=with_attentions,
        reduction=reduction,
        attention_type=attention_type
    )
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (3, 2048, 7, 7)

    # for RX50_32x4d
    model = ResNetDBackbone(
        layer_blocks=(3, 4, 6, 3),
        groups=32,
        width_per_group=4,
        with_attentions=with_attentions,
        reduction=reduction,
        attention_type=attention_type,
        block_layer=Bottleneck
    )
    print(model)

    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (3, 2048, 7, 7)


if __name__ == '__main__':
    print('*' * 10 + ' resnetd backbone')
    test_resnet_d_backbone()
    print('*' * 10 + ' se resnetd bottleneck')
    test_attention_resnet_d_backbone(with_attentions=(1, 1, 1, 1),
                                     reduction=16,
                                     attention_type='SqueezeAndExcitationBlock2D')
    print('*' * 10 + ' nl resnetd bottleneck')
    test_attention_resnet_d_backbone(with_attentions=(0, (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), 0),
                                     reduction=16,
                                     attention_type='NonLocal2DEmbeddedGaussian')
    print('*' * 10 + ' snl resnetd bottleneck')
    test_attention_resnet_d_backbone(with_attentions=(0, (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), 0),
                                     reduction=16,
                                     attention_type='SimplifiedNonLocal2DEmbeddedGaussian')
    print('*' * 10 + ' gc resnetd bottleneck')
    test_attention_resnet_d_backbone(with_attentions=(0, 1, 1, 0),
                                     reduction=4,
                                     attention_type='GlobalContextBlock2D')

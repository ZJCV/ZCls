# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午4:16
@file: test_resnet.py
@author: zj
@description: 
"""

import torch

from zcls.model.recognizers.attention_resnet_recognizer import AttentionResNetRecognizer


def test_attention_resnet():
    # gc
    model = AttentionResNetRecognizer(
        arch="resnet50",
        feature_dims=2048,
        num_classes=1000,
        with_attention=(1, 1, 1, 1),
        reduction=16,
        attention_type='GlobalContextBlock2D'
    )
    print(model)

    data = torch.randn(10, 3, 224, 224)
    outputs = model(data)['probs']
    print(outputs.shape)

    assert outputs.shape == (10, 1000)

    # snl
    model = AttentionResNetRecognizer(
        arch="resnet50",
        feature_dims=2048,
        num_classes=1000,
        with_attention=(0, (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), 0),
        attention_type='SimplifiedNonLocal2DEmbeddedGaussian'
    )
    print(model)

    data = torch.randn(10, 3, 224, 224)
    outputs = model(data)['probs']
    print(outputs.shape)

    assert outputs.shape == (10, 1000)

    # nl
    model = AttentionResNetRecognizer(
        arch="resnet50",
        feature_dims=2048,
        num_classes=1000,
        with_attention=(0, (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), 0),
        attention_type='NonLocal2DEmbeddedGaussian',
    )
    print(model)

    data = torch.randn(10, 3, 224, 224)
    outputs = model(data)['probs']
    print(outputs.shape)

    assert outputs.shape == (10, 1000)

    # se
    model = AttentionResNetRecognizer(
        arch="resnet50",
        feature_dims=2048,
        num_classes=1000,
        with_attention=(1, 1, 1, 1),
        reduction=16,
        attention_type='SqueezeAndExcitationBlock2D'
    )
    print(model)

    data = torch.randn(10, 3, 224, 224)
    outputs = model(data)['probs']
    print(outputs.shape)

    assert outputs.shape == (10, 1000)


def test_attention_resnetxt():
    groups = 32
    width_per_group = 4

    # gc
    model = AttentionResNetRecognizer(
        arch="resnet50",
        feature_dims=2048,
        num_classes=1000,
        with_attention=(1, 1, 1, 1),
        reduction=16,
        attention_type='GlobalContextBlock2D',
        groups=groups,
        width_per_group=width_per_group
    )
    print(model)

    data = torch.randn(10, 3, 224, 224)
    outputs = model(data)['probs']
    print(outputs.shape)

    assert outputs.shape == (10, 1000)

    # snl
    model = AttentionResNetRecognizer(
        arch="resnet50",
        feature_dims=2048,
        num_classes=1000,
        with_attention=(0, (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), 0),
        attention_type='SimplifiedNonLocal2DEmbeddedGaussian',
        groups=groups,
        width_per_group=width_per_group
    )
    print(model)

    data = torch.randn(10, 3, 224, 224)
    outputs = model(data)['probs']
    print(outputs.shape)

    assert outputs.shape == (10, 1000)

    # nl
    model = AttentionResNetRecognizer(
        arch="resnet50",
        feature_dims=2048,
        num_classes=1000,
        with_attention=(0, (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), 0),
        attention_type='NonLocal2DEmbeddedGaussian',
        groups=groups,
        width_per_group=width_per_group
    )
    print(model)

    data = torch.randn(10, 3, 224, 224)
    outputs = model(data)['probs']
    print(outputs.shape)

    assert outputs.shape == (10, 1000)

    # se
    model = AttentionResNetRecognizer(
        arch="resnet50",
        feature_dims=2048,
        num_classes=1000,
        with_attention=(1, 1, 1, 1),
        reduction=16,
        attention_type='SqueezeAndExcitationBlock2D',
        groups=groups,
        width_per_group=width_per_group
    )
    print(model)

    data = torch.randn(10, 3, 224, 224)
    outputs = model(data)['probs']
    print(outputs.shape)

    assert outputs.shape == (10, 1000)


if __name__ == '__main__':
    test_attention_resnet()
    test_attention_resnetxt()

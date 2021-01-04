# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午4:16
@file: test_resnet.py
@author: zj
@description: 
"""

import torch

from zcls.config import cfg
from zcls.config.key_word import KEY_OUTPUT
from zcls.model.norm_helper import get_norm
from zcls.model.recognizers.resnet_recognizer import TorchvisionResNet, ResNetRecognizer, build_resnet


def test_data(model, input_shape, output_shape):
    data = torch.randn(input_shape)
    outputs = model(data)[KEY_OUTPUT]
    print(outputs.shape)

    assert outputs.shape == output_shape


def test_resnet():
    # for torchvision
    model = TorchvisionResNet(
        arch='resnet50',
        num_classes=1000
    )
    print(model)
    test_data(model, (3, 3, 224, 224), (3, 1000))

    # for custom
    model = ResNetRecognizer(
        arch="resnet50",
        num_classes=1000
    )
    print(model)
    test_data(model, (3, 3, 224, 224), (3, 1000))

    # resnetxt_32x4d
    model = ResNetRecognizer(
        arch="resnext50_32x4d",
        num_classes=1000
    )
    print(model)
    test_data(model, (3, 3, 224, 224), (3, 1000))


def test_resnet_gn():
    cfg.MODEL.NORM.TYPE = 'GroupNorm'
    norm_layer = get_norm(cfg)
    print(norm_layer)

    # for custom
    model = ResNetRecognizer(
        arch="resnet50",
        num_classes=1000,
        norm_layer=norm_layer
    )
    print(model)
    test_data(model, (1, 3, 224, 224), (1, 1000))

    # resnetxt_32x4d
    model = ResNetRecognizer(
        arch="resnext50_32x4d",
        num_classes=1000,
        norm_layer=norm_layer
    )
    print(model)
    test_data(model, (3, 3, 224, 224), (3, 1000))


def test_config():
    config_file = 'configs/benchmarks/r50_custom_cifar100_224_e100_rmsprop.yaml'
    cfg.merge_from_file(config_file)

    model = build_resnet(cfg)
    print(model)
    test_data(model, (1, 3, 224, 224), (1, 100))

    config_file = 'configs/benchmarks/r50_torchvision_cifar100_224_e100_rmsprop.yaml'
    cfg.merge_from_file(config_file)

    model = build_resnet(cfg)
    print(model)
    test_data(model, (1, 3, 224, 224), (1, 100))


def test_attention_resnet(
        with_attentions=(1, 1, 1, 1),
        reduction=16,
        attention_type='SqueezeAndExcitationBlock2D'
):
    # for custom
    model = ResNetRecognizer(
        arch="resnet50",
        with_attentions=with_attentions,
        reduction=reduction,
        attention_type=attention_type,
        num_classes=1000
    )
    print(model)
    test_data(model, (3, 3, 224, 224), (3, 1000))

    # resnetxt_32x4d
    model = ResNetRecognizer(
        arch="resnext50_32x4d",
        with_attentions=with_attentions,
        reduction=reduction,
        attention_type=attention_type,
        num_classes=1000
    )
    print(model)
    test_data(model, (3, 3, 224, 224), (3, 1000))


def test_attention_resnetd(
        with_attentions=(1, 1, 1, 1),
        reduction=16,
        attention_type='SqueezeAndExcitationBlock2D'
):
    # for custom
    model = ResNetRecognizer(
        arch="resnetd50",
        with_attentions=with_attentions,
        reduction=reduction,
        attention_type=attention_type,
        num_classes=1000
    )
    print(model)
    test_data(model, (3, 3, 224, 224), (3, 1000))

    # resnetxt_32x4d
    model = ResNetRecognizer(
        arch="resnedxt50_32x4d",
        with_attentions=with_attentions,
        reduction=reduction,
        attention_type=attention_type,
        num_classes=1000
    )
    print(model)
    test_data(model, (3, 3, 224, 224), (3, 1000))


def test_sknet():
    # resnetd
    model = ResNetRecognizer(
        arch="sknet50",
        num_classes=1000
    )
    print(model)
    test_data(model, (3, 3, 224, 224), (3, 1000))


def test_resnetst():
    # resnetd
    model = ResNetRecognizer(
        arch="resnetst50_2s2x40d",
        num_classes=1000
    )
    print(model)
    test_data(model, (3, 3, 224, 224), (3, 1000))

    # resnetd
    model = ResNetRecognizer(
        arch="resnetst50_2s2x40d_fast",
        num_classes=1000
    )
    print(model)
    test_data(model, (3, 3, 224, 224), (3, 1000))


if __name__ == '__main__':
    print('*' * 10 + ' resnet')
    test_resnet()
    print('*' * 10 + ' resnet gn')
    test_resnet_gn()
    print('*' * 10 + ' se resnet')
    test_attention_resnet(with_attentions=(1, 1, 1, 1),
                          reduction=16,
                          attention_type='SqueezeAndExcitationBlock2D')
    print('*' * 10 + ' nl resnet')
    test_attention_resnet(with_attentions=(0, (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), 0),
                          reduction=16,
                          attention_type='NonLocal2DEmbeddedGaussian')
    print('*' * 10 + ' snl resnet')
    test_attention_resnet(with_attentions=(0, (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), 0),
                          reduction=16,
                          attention_type='SimplifiedNonLocal2DEmbeddedGaussian')
    print('*' * 10 + ' gc resnet')
    test_attention_resnet(with_attentions=(0, 1, 1, 1),
                          reduction=16,
                          attention_type='GlobalContextBlock2D')

    print('*' * 10 + ' se resnetd')
    test_attention_resnetd(with_attentions=(1, 1, 1, 1),
                           reduction=16,
                           attention_type='SqueezeAndExcitationBlock2D')
    print('*' * 10 + ' nl resnetd')
    test_attention_resnetd(with_attentions=(0, (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), 0),
                           reduction=16,
                           attention_type='NonLocal2DEmbeddedGaussian')
    print('*' * 10 + ' snl resnetd')
    test_attention_resnetd(with_attentions=(0, (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), 0),
                           reduction=16,
                           attention_type='SimplifiedNonLocal2DEmbeddedGaussian')
    print('*' * 10 + ' gc resnetd')
    test_attention_resnetd(with_attentions=(0, 1, 1, 1),
                           reduction=16,
                           attention_type='GlobalContextBlock2D')

    print('*' * 10 + ' sknet')
    test_sknet()

    print('*' * 10 + ' resnetst')
    test_resnetst()

    test_config()

# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午4:16
@file: test_resnest.py
@author: zj
@description: 
"""

import torch

from zcls.config import cfg
from zcls.config.key_word import KEY_OUTPUT
from zcls.model.recognizers.resnet.resnet import ResNet
from zcls.model.recognizers.resnet.torchvision_resnet import build_torchvision_resnet


def test_data(model, input_shape, output_shape):
    data = torch.randn(input_shape)
    outputs = model(data)[KEY_OUTPUT]
    print(outputs.shape)

    assert outputs.shape == output_shape


def test_resnet():
    config_file = 'configs/benchmarks/resnet/r50_cifar100_224_e100_rmsprop.yaml'
    cfg.merge_from_file(config_file)

    model = ResNet(cfg)
    print(model)
    test_data(model, (1, 3, 224, 224), (1, 100))

    config_file = 'configs/benchmarks/resnet/rxt50_32x4d_cifar100_224_e100_rmsprop.yaml'
    cfg.merge_from_file(config_file)

    model = ResNet(cfg)
    print(model)
    test_data(model, (1, 3, 224, 224), (1, 100))

    config_file = 'configs/benchmarks/resnet/r50_torchvision_cifar100_224_e100_rmsprop.yaml'
    cfg.merge_from_file(config_file)

    model = build_torchvision_resnet(cfg)
    print(model)
    test_data(model, (1, 3, 224, 224), (1, 100))

    config_file = 'configs/benchmarks/resnet/rxt50_32x4d_torchvision_cifar100_224_e100_rmsprop.yaml'
    cfg.merge_from_file(config_file)

    model = build_torchvision_resnet(cfg)
    print(model)
    test_data(model, (1, 3, 224, 224), (1, 100))


def test_resnetd():
    config_file = 'configs/benchmarks/resnet/rd50_cifar100_224_e100_rmsprop.yaml'
    cfg.merge_from_file(config_file)

    model = ResNet(cfg)
    print(model)
    test_data(model, (1, 3, 224, 224), (1, 100))

    config_file = 'configs/benchmarks/resnet/rxtd50_32x4d_cifar100_224_e100_rmsprop.yaml'
    cfg.merge_from_file(config_file)

    model = ResNet(cfg)
    print(model)
    test_data(model, (1, 3, 224, 224), (1, 100))

    config_file = 'configs/benchmarks/resnet/rxtd50_32x4d_fast_avg_cifar100_224_e100_rmsprop.yaml'
    cfg.merge_from_file(config_file)

    model = ResNet(cfg)
    print(model)
    test_data(model, (1, 3, 224, 224), (1, 100))

    config_file = 'configs/benchmarks/resnet/rxtd50_32x4d_avg_cifar100_224_e100_rmsprop.yaml'
    cfg.merge_from_file(config_file)

    model = ResNet(cfg)
    print(model)
    test_data(model, (1, 3, 224, 224), (1, 100))


def test_sknet():
    config_file = 'configs/benchmarks/resnet/sknet50_cifar100_224_e100_rmsprop.yaml'
    cfg.merge_from_file(config_file)

    model = ResNet(cfg)
    print(model)
    test_data(model, (3, 3, 224, 224), (3, 100))


def test_resnest():
    config_file = 'configs/benchmarks/resnet/rstd50_2s2x40d_cifar100_224_e100_rmsprop.yaml'
    cfg.merge_from_file(config_file)

    model = ResNet(cfg)
    print(model)
    test_data(model, (3, 3, 224, 224), (3, 100))

    config_file = 'configs/benchmarks/resnet/rstd50_2s2x40d_fast_cifar100_224_e100_rmsprop.yaml'
    cfg.merge_from_file(config_file)

    model = ResNet(cfg)
    print(model)
    test_data(model, (3, 3, 224, 224), (3, 100))

    config_file = 'configs/benchmarks/resnet/rstd50_2s2x40d_fast_official_cifar100_224_e100_rmsprop.yaml'
    cfg.merge_from_file(config_file)

    model = ResNet(cfg)
    print(model)
    test_data(model, (3, 3, 224, 224), (3, 100))


if __name__ == '__main__':
    print('*' * 10 + ' resnet')
    test_resnet()
    print('*' * 10 + ' resnetd')
    test_resnetd()
    print('*' * 10 + ' sknet')
    test_sknet()
    print('*' * 10 + ' resnest')
    test_resnest()

    # print('*' * 10 + ' se resnet')
    # test_attention_resnet(with_attentions=(1, 1, 1, 1),
    #                       reduction=16,
    #                       attention_type='SqueezeAndExcitationBlock2D')
    # print('*' * 10 + ' nl resnet')
    # test_attention_resnet(with_attentions=(0, (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), 0),
    #                       reduction=16,
    #                       attention_type='NonLocal2DEmbeddedGaussian')
    # print('*' * 10 + ' snl resnet')
    # test_attention_resnet(with_attentions=(0, (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), 0),
    #                       reduction=16,
    #                       attention_type='SimplifiedNonLocal2DEmbeddedGaussian')
    # print('*' * 10 + ' gc resnet')
    # test_attention_resnet(with_attentions=(0, 1, 1, 1),
    #                       reduction=16,
    #                       attention_type='GlobalContextBlock2D')
    #
    # print('*' * 10 + ' se resnetd')
    # test_attention_resnetd(with_attentions=(1, 1, 1, 1),
    #                        reduction=16,
    #                        attention_type='SqueezeAndExcitationBlock2D')
    # print('*' * 10 + ' nl resnetd')
    # test_attention_resnetd(with_attentions=(0, (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), 0),
    #                        reduction=16,
    #                        attention_type='NonLocal2DEmbeddedGaussian')
    # print('*' * 10 + ' snl resnetd')
    # test_attention_resnetd(with_attentions=(0, (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), 0),
    #                        reduction=16,
    #                        attention_type='SimplifiedNonLocal2DEmbeddedGaussian')
    # print('*' * 10 + ' gc resnetd')
    # test_attention_resnetd(with_attentions=(0, 1, 1, 1),
    #                        reduction=16,
    #                        attention_type='GlobalContextBlock2D')

# -*- coding: utf-8 -*-

"""
@date: 2020/12/30 下午9:43
@file: test_mobilenet_v3_recognizer.py
@author: zj
@description: 
"""

import torch

from zcls.config import cfg
from zcls.config.key_word import KEY_OUTPUT
from zcls.model.recognizers.mobilenet.mobilenetv3_recognizer import MobileNetV3Recognizer, build_mobilenet_v3


def test_mobilenet_v3():
    data = torch.randn(1, 3, 224, 224)
    model = MobileNetV3Recognizer(
        arch='mobilenetv3-large',
        in_planes=3,
        width_multiplier=1.,
        round_nearest=8,
        reduction=4,
        attention_type='SqueezeAndExcitationBlock2D',
        num_classes=1000,
        zcls_pretrained="",
        pretrained_num_classes=1000,
        fix_bn=False,
        partial_bn=False,
        conv_layer=None,
        norm_layer=None,
        act_layer=None
    )
    print(model)

    outputs = model(data)[KEY_OUTPUT]
    print(outputs.shape)

    assert outputs.shape == (1, 1000)

    model = MobileNetV3Recognizer(
        arch='mobilenetv3-small',
        in_planes=3,
        width_multiplier=1.,
        round_nearest=8,
        reduction=4,
        attention_type='SqueezeAndExcitationBlock2D',
        num_classes=1000,
        zcls_pretrained="",
        pretrained_num_classes=1000,
        fix_bn=False,
        partial_bn=False,
        conv_layer=None,
        norm_layer=None,
        act_layer=None
    )
    print(model)

    outputs = model(data)[KEY_OUTPUT]
    print(outputs.shape)

    assert outputs.shape == (1, 1000)


def test_config():
    data = torch.randn(1, 3, 224, 224)

    config_file = 'configs/benchmarks/mbv3_large_se_custom_cifar100_224_e50.yaml'
    cfg.merge_from_file(config_file)
    model = build_mobilenet_v3(cfg)
    print(model)
    outputs = model(data)
    outputs = model(data)[KEY_OUTPUT]
    print(outputs.shape)

    assert outputs.shape == (1, 100)

    config_file = 'configs/benchmarks/mbv3_large_se_hsigmoid_custom_cifar100_224_e50.yaml'
    cfg.merge_from_file(config_file)
    model = build_mobilenet_v3(cfg)
    print(model)
    outputs = model(data)
    outputs = model(data)[KEY_OUTPUT]
    print(outputs.shape)

    assert outputs.shape == (1, 100)

    config_file = 'configs/benchmarks/mbv3_large_custom_cifar100_224_e50.yaml'
    cfg.merge_from_file(config_file)
    model = build_mobilenet_v3(cfg)
    print(model)
    outputs = model(data)
    outputs = model(data)[KEY_OUTPUT]
    print(outputs.shape)

    assert outputs.shape == (1, 100)

    config_file = 'configs/benchmarks/mbv3_small_se_custom_cifar100_224_e50.yaml'
    cfg.merge_from_file(config_file)
    model = build_mobilenet_v3(cfg)
    print(model)
    outputs = model(data)
    outputs = model(data)[KEY_OUTPUT]
    print(outputs.shape)

    assert outputs.shape == (1, 100)

    config_file = 'configs/benchmarks/mbv3_small_se_hsigmoid_custom_cifar100_224_e50.yaml'
    cfg.merge_from_file(config_file)
    model = build_mobilenet_v3(cfg)
    print(model)
    outputs = model(data)
    outputs = model(data)[KEY_OUTPUT]
    print(outputs.shape)

    assert outputs.shape == (1, 100)

    config_file = 'configs/benchmarks/mbv3_small_custom_cifar100_224_e50.yaml'
    cfg.merge_from_file(config_file)
    model = build_mobilenet_v3(cfg)
    print(model)
    outputs = model(data)
    outputs = model(data)[KEY_OUTPUT]
    print(outputs.shape)

    assert outputs.shape == (1, 100)


if __name__ == '__main__':
    # test_mobilenet_v3()
    test_config()

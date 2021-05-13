# -*- coding: utf-8 -*-

"""
@date: 2020/12/30 下午9:43
@file: test_mobilenet_v3.py
@author: zj
@description: 
"""

import torch

from zcls.config import cfg
from zcls.config.key_word import KEY_OUTPUT
from zcls.model.recognizers.mobilenet.mobilenetv3 import MobileNetV3


def test_config():
    data = torch.randn(1, 3, 224, 224)

    config_file = 'configs/cifar/mbv3_large_cifar100_224_e100_sgd.yaml'
    cfg.merge_from_file(config_file)
    model = MobileNetV3(cfg)
    print(model)
    outputs = model(data)
    outputs = model(data)[KEY_OUTPUT]
    print(outputs.shape)

    assert outputs.shape == (1, 100)

    config_file = 'configs/cifar/mbv3_large_se_cifar100_224_e100_sgd.yaml'
    cfg.merge_from_file(config_file)
    model = MobileNetV3(cfg)
    print(model)
    outputs = model(data)
    outputs = model(data)[KEY_OUTPUT]
    print(outputs.shape)

    assert outputs.shape == (1, 100)

    config_file = 'configs/cifar/mbv3_large_se_hsigmoid_cifar100_224_e100.yaml'
    cfg.merge_from_file(config_file)
    model = MobileNetV3(cfg)
    print(model)
    outputs = model(data)
    outputs = model(data)[KEY_OUTPUT]
    print(outputs.shape)

    assert outputs.shape == (1, 100)

    config_file = 'configs/cifar/mbv3_small_cifar100_224_e100_sgd.yaml'
    cfg.merge_from_file(config_file)
    model = MobileNetV3(cfg)
    print(model)
    outputs = model(data)
    outputs = model(data)[KEY_OUTPUT]
    print(outputs.shape)

    assert outputs.shape == (1, 100)

    config_file = 'configs/cifar/mbv3_small_se_cifar100_224_e100.yaml'
    cfg.merge_from_file(config_file)
    model = MobileNetV3(cfg)
    print(model)
    outputs = model(data)
    outputs = model(data)[KEY_OUTPUT]
    print(outputs.shape)

    assert outputs.shape == (1, 100)

    config_file = 'configs/cifar/mbv3_small_se_hsigmoid_cifar100_224_e100.yaml'
    cfg.merge_from_file(config_file)
    model = MobileNetV3(cfg)
    print(model)
    outputs = model(data)
    outputs = model(data)[KEY_OUTPUT]
    print(outputs.shape)

    assert outputs.shape == (1, 100)


if __name__ == '__main__':
    test_config()

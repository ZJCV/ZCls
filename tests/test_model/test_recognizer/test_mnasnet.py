# -*- coding: utf-8 -*-

"""
@date: 2020/12/3 下午8:27
@file: test_mobilenet_v1.py
@author: zj
@description: 
"""

import torch

from zcls.config import cfg
from zcls.config.key_word import KEY_OUTPUT
from zcls.model.recognizers.mobilenet.mnasnet import MNASNet
from zcls.model.recognizers.mobilenet.torchvision_mnasnet import build_torchvision_mnasnet


def test_mnasnet():
    cfg.merge_from_file('configs/benchmarks/lightweight/mnasnet_a1_1_3_cifar100_224_e100.yaml')
    data = torch.randn(1, 3, 224, 224)

    for wm in [0.5, 0.75, 1.0, 1.3]:
        cfg.MODEL.COMPRESSION.WIDTH_MULTIPLIER = wm
        model = MNASNet(cfg)
        print(model)

        outputs = model(data)[KEY_OUTPUT]
        print(outputs.shape)

        assert outputs.shape == (1, 100)

    cfg.merge_from_file('configs/benchmarks/lightweight/mnasnet_b1_1_3_torchvision_cifar100_224_e100_sgd.yaml')
    for wm in [0.5, 0.75, 1.0, 1.3]:
        cfg.MODEL.COMPRESSION.WIDTH_MULTIPLIER = wm
        model = build_torchvision_mnasnet(cfg)
        print(model)

        outputs = model(data)[KEY_OUTPUT]
        print(outputs.shape)

        assert outputs.shape == (1, 100)


def test_config():
    data = torch.randn(1, 3, 224, 224)

    config_file = 'configs/benchmarks/lightweight/mnasnet_a1_1_3_cifar100_224_e100.yaml'
    cfg.merge_from_file(config_file)
    model = MNASNet(cfg)
    print(model)
    outputs = model(data)[KEY_OUTPUT]
    print(outputs.shape)

    assert outputs.shape == (1, 100)

    config_file = 'configs/benchmarks/lightweight/mnasnet_a1_1_3_se_cifar100_224_e100.yaml'
    cfg.merge_from_file(config_file)
    model = MNASNet(cfg)
    print(model)
    outputs = model(data)[KEY_OUTPUT]
    print(outputs.shape)

    assert outputs.shape == (1, 100)

    config_file = 'configs/benchmarks/lightweight/mnasnet_b1_1_3_cifar100_224_e100_sgd.yaml'
    cfg.merge_from_file(config_file)
    model = MNASNet(cfg)
    print(model)
    outputs = model(data)[KEY_OUTPUT]
    print(outputs.shape)

    assert outputs.shape == (1, 100)

    config_file = 'configs/benchmarks/lightweight/mnasnet_b1_1_3_torchvision_cifar100_224_e100_sgd.yaml'
    cfg.merge_from_file(config_file)
    model = build_torchvision_mnasnet(cfg)
    print(model)
    outputs = model(data)[KEY_OUTPUT]
    print(outputs.shape)

    assert outputs.shape == (1, 100)


if __name__ == '__main__':
    test_mnasnet()
    test_config()

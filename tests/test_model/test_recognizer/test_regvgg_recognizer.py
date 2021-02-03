# -*- coding: utf-8 -*-

"""
@date: 2021/2/2 下午5:46
@file: test_regvgg_recognizer.py
@author: zj
@description: 
"""

import torch

from zcls.config import cfg
from zcls.config.key_word import KEY_OUTPUT
from zcls.model.recognizers.build import build_recognizer
from zcls.model.recognizers.repvgg_recognizer import RepVGGRecognizer
from zcls.model.recognizers.repvgg_recognizer import arch_settings
from zcls.model.conv_helper import fuse_repvgg_block, fuse_acblock


def test_regvgg_recognizer():
    data = torch.randn(1, 3, 224, 224)
    for key in arch_settings.keys():
        model = RepVGGRecognizer(arch=key)
        print(model)
        outputs = model(data)[KEY_OUTPUT]
        assert outputs.shape == (1, 1000)


def test_config_file():
    data = torch.randn(3, 3, 224, 224)

    print('repvgg_plain_custom_cifar100_224_e100_sgd')
    config_file = "configs/benchmarks/repvgg/repvgg_plain_custom_cifar100_224_e100_sgd.yaml"
    cfg.merge_from_file(config_file)

    device = torch.device('cpu')
    model = build_recognizer(cfg, device)
    print(model)
    outputs = model(data)[KEY_OUTPUT]

    assert outputs.shape == (3, 100)

    print('repvgg_b2g4_custom_cifar100_224_e100_sgd')
    config_file = "configs/benchmarks/repvgg/repvgg_b2g4_custom_cifar100_224_e100_sgd.yaml"
    cfg.merge_from_file(config_file)

    device = torch.device('cpu')
    model = build_recognizer(cfg, device)
    print(model)
    outputs = model(data)[KEY_OUTPUT]

    assert outputs.shape == (3, 100)
    fuse_repvgg_block(model)
    print(model)
    outputs = model(data)[KEY_OUTPUT]

    assert outputs.shape == (3, 100)

    print('repvgg_b2g4_acb_custom_cifar100_224_e100_sgd')
    config_file = "configs/benchmarks/repvgg/repvgg_b2g4_acb_custom_cifar100_224_e100_sgd.yaml"
    cfg.merge_from_file(config_file)

    device = torch.device('cpu')
    model = build_recognizer(cfg, device)
    print(model)
    outputs = model(data)[KEY_OUTPUT]

    assert outputs.shape == (3, 100)
    # 注意：如果在RepVGG中嵌入了ACBlock，融合时应该先acb再regvgg
    fuse_acblock(model)
    print(model)
    fuse_repvgg_block(model)
    print(model)
    outputs = model(data)[KEY_OUTPUT]

    assert outputs.shape == (3, 100)


if __name__ == '__main__':
    # test_regvgg_recognizer()
    test_config_file()

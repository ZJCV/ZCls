# -*- coding: utf-8 -*-

"""
@date: 2020/12/29 下午5:04
@file: test_shufflenetv2.py
@author: zj
@description: 
"""

import torch

from zcls.config import cfg
from zcls.config.key_word import KEY_OUTPUT
from zcls.model.recognizers.shufflenet.shufflenetv2 import ShuffleNetV2
from zcls.model.recognizers.shufflenet.torchvision_sfv2 import build_torchvision_sfv2


def test_data(model):
    data = torch.randn(1, 3, 224, 224)
    outputs = model(data)[KEY_OUTPUT]
    print(outputs.shape)

    assert outputs.shape == (1, 100)


def test_shufflenet_v2():
    cfg.merge_from_file('configs/benchmarks/lightweight/sfv2_x2_0_cifar100_224_e100.yaml')
    print(cfg)
    model = ShuffleNetV2(cfg)
    print(model)

    test_data(model)

    cfg.merge_from_file('configs/benchmarks/lightweight/sfv2_torchvision_cifar100_224_e100.yaml')
    model = build_torchvision_sfv2(cfg)
    print(model)
    test_data(model)


if __name__ == '__main__':
    test_shufflenet_v2()

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


def test_data(model, input_shape, output_shape):
    data = torch.randn(input_shape)
    outputs = model(data)[KEY_OUTPUT]
    print(outputs.shape)

    assert outputs.shape == output_shape


def test_sknet():
    config_file = 'configs/benchmarks/resnet/sknet50_zcls_imagenet_224.yaml'
    cfg.merge_from_file(config_file)

    model = ResNet(cfg)
    print(model)
    test_data(model, (3, 3, 224, 224), (3, 1000))


if __name__ == '__main__':
    print('*' * 10 + ' sknet')
    test_sknet()

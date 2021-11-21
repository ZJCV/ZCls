# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午4:16
@file: test_resnest.py
@author: zj
@description: 
"""

import torch

from zcls.config.key_word import KEY_OUTPUT
from zcls.config import cfg
from zcls.model.recognizers.resnet.official_resnest import OfficialResNeSt
from zcls.model.recognizers.build import build_recognizer


def test_data(model, input_shape, output_shape):
    data = torch.randn(input_shape)
    outputs = model(data)[KEY_OUTPUT]
    print(outputs.shape)

    assert outputs.shape == output_shape


def test_official_resnest():
    # resnetd
    model = OfficialResNeSt(
        arch="resnest50_2s2x40d",
        num_classes=1000
    )
    print(model)
    test_data(model, (3, 3, 224, 224), (3, 1000))

    # resnetd
    model = OfficialResNeSt(
        arch="resnest50_fast_2s2x40d",
        num_classes=1000
    )
    print(model)
    test_data(model, (3, 3, 224, 224), (3, 1000))


def test_zcls_resnest():
    cfg.merge_from_file('configs/benchmarks/resnet-resnext/resnest50_fast_2s1x64d_imagenet_224.yaml')
    model = build_recognizer(cfg, torch.device('cpu'))
    print(model)
    test_data(model, (3, 3, 224, 224), (3, 1000))


if __name__ == '__main__':
    print('*' * 10 + ' resnetst')
    # test_official_resnest()
    test_zcls_resnest()

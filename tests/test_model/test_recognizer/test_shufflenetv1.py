# -*- coding: utf-8 -*-

"""
@date: 2021/5/16 下午10:22
@file: test_shufflenetv1.py
@author: zj
@description: 
"""

import torch

from zcls.config import cfg
from zcls.config.key_word import KEY_OUTPUT
from zcls.model.recognizers.build import build_recognizer


def test_data(model):
    data = torch.ones(1, 3, 224, 224)
    outputs = model(data)[KEY_OUTPUT]
    print(outputs.shape)

    # print(outputs)

    assert outputs.shape == (1, 1000)


def test_shufflenet():
    cfg.merge_from_file('configs/benchmarks/shufflenet/shufflenet_v1_3g1x_zcls_imagenet_224.yaml')
    # print(cfg)
    model = build_recognizer(cfg, torch.device('cpu'))
    # print(model)

    test_data(model)


if __name__ == '__main__':
    test_shufflenet()

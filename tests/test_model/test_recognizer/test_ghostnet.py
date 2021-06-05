# -*- coding: utf-8 -*-

"""
@date: 2021/6/4 下午5:27
@file: test_ghostnet.py
@author: zj
@description: 
"""

import torch

from zcls.config import cfg
from zcls.config.key_word import KEY_OUTPUT
from zcls.model.recognizers.ghostnet.ghostnet import GhostNet


def test_data(model):
    data = torch.randn(1, 3, 224, 224)
    outputs = model(data)[KEY_OUTPUT]
    print(outputs.shape)

    assert outputs.shape == (1, 1000)


def test_shufflenet_v2():
    cfg.merge_from_file('configs/benchmarks/ghostnet/ghostnet_x1_0_zcls_imagenet_224.yaml')
    print(cfg)
    model = GhostNet(cfg)
    print(model)

    test_data(model)


if __name__ == '__main__':
    test_shufflenet_v2()

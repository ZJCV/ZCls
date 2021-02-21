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
from zcls.model.backbones.misc import make_divisible
from zcls.model.recognizers.mobilenet.mobilenetv1 import build_mobilenet_v1
from zcls.model.recognizers.mobilenet.mobilenetv1 import MobileNetV1


def test_mobilenetv1():
    for s in [224, 192, 160, 128]:
        for wm in [1.0, 0.75, 0.5, 0.25]:
            cfg.merge_from_file('configs/benchmarks/lightweight/mbv1_cifar100_224_e100.yaml')
            cfg.MODEL.COMPRESSION.WIDTH_MULTIPLIER = wm
            round_nearest = cfg.MODEL.COMPRESSION.ROUND_NEAREST
            feature_dims = make_divisible(cfg.MODEL.HEAD.FEATURE_DIMS * wm, round_nearest)
            cfg.MODEL.HEAD.FEATURE_DIMS = feature_dims

            print(f's: {s}, wn: {wm}, feature_dims: {feature_dims}')
            model = MobileNetV1(cfg)
            # print(model)

            data = torch.randn(1, 3, s, s)
            outputs = model(data)[KEY_OUTPUT]
            print(outputs.shape)

            assert outputs.shape == (1, 100)


if __name__ == '__main__':
    test_mobilenetv1()

# -*- coding: utf-8 -*-

"""
@date: 2020/12/3 下午8:27
@file: test_mobilenetv1_recognizer.py
@author: zj
@description: 
"""

import torch

from zcls.config import cfg
from zcls.config.key_word import KEY_OUTPUT
from zcls.model.recognizers.mobilenet.mobilenetv1 import build_mobilenet_v1
from zcls.model.recognizers.mobilenet.mobilenetv1 import MobileNetV1Recognizer


def test_mobilenetv1():
    for s in [224, 192, 160, 128]:
        for wm in [1.0, 0.75, 0.5, 0.25]:
            print(f's: {s}, wn: {wm}')
            model = MobileNetV1Recognizer(
                width_multiplier=wm
            )
            # print(model)

            data = torch.randn(1, 3, s, s)
            outputs = model(data)[KEY_OUTPUT]
            print(outputs.shape)

            assert outputs.shape == (1, 1000)


def test_config():
    config_file = "configs/benchmarks/mbv1_custom_cifar100_224.yaml"
    cfg.merge_from_file(config_file)

    model = build_mobilenet_v1(cfg)
    print(model)


if __name__ == '__main__':
    # test_mobilenetv1()
    test_config()

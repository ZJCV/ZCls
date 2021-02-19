# -*- coding: utf-8 -*-

"""
@date: 2020/12/3 下午8:27
@file: test_mobilenetv1_recognizer.py
@author: zj
@description: 
"""

import torch

from zcls.config.key_word import KEY_OUTPUT
from zcls.model.recognizers.mobilenet.mobilenetv2_recognizer import MobileNetV2Recognizer


def test_mobilenet_v2():
    for s in [224, 192, 160, 128]:
        for wm in [1.0, 0.75, 0.5, 0.25]:
            print(f's: {s}, wn: {wm}')
            model = MobileNetV2Recognizer(
                width_multiplier=wm
            )
            # print(model)

            data = torch.randn(1, 3, s, s)
            outputs = model(data)[KEY_OUTPUT]
            print(outputs.shape)

            assert outputs.shape == (1, 1000)


if __name__ == '__main__':
    test_mobilenet_v2()

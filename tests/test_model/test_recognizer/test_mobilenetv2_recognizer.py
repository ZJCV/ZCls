# -*- coding: utf-8 -*-

"""
@date: 2020/12/3 下午8:27
@file: test_mobilenetv1_recognizer.py
@author: zj
@description: 
"""

import torch

from zcls.model.recognizers.mobilenetv2_recognizer import MobileNetV2Recognizer


def test_mobilenetv1():
    for s in [224, 192, 160, 128]:
        for wm in [1.0, 0.75, 0.5, 0.25]:
            print(f's: {s}, wn: {wm}')
            model = MobileNetV2Recognizer(
                width_multiplier=wm
            )
            # print(model)

            data = torch.randn(1, 3, s, s)
            outputs = model(data)['probs']
            print(outputs.shape)

            assert outputs.shape == (1, 1000)


if __name__ == '__main__':
    test_mobilenetv1()

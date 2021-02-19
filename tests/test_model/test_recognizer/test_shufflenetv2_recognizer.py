# -*- coding: utf-8 -*-

"""
@date: 2020/12/29 下午5:04
@file: test_shufflenetv2_recognizer.py
@author: zj
@description: 
"""

import torch

from zcls.config.key_word import KEY_OUTPUT
from zcls.model.recognizers.shufflenet.shufflenetv2 import ShuffleNetV2Recognizer, TorchvisionShuffleNetV2


def test_data(model):
    data = torch.randn(1, 3, 224, 224)
    outputs = model(data)[KEY_OUTPUT]
    print(outputs.shape)

    assert outputs.shape == (1, 1000)


def test_shufflenet_v2():
    model = ShuffleNetV2Recognizer()
    print(model)

    test_data(model)

    model = TorchvisionShuffleNetV2()
    print(model)
    test_data(model)


if __name__ == '__main__':
    test_shufflenet_v2()

# -*- coding: utf-8 -*-

"""
@date: 2020/12/3 下午8:27
@file: test_mobilenetv1_recognizer.py
@author: zj
@description: 
"""

import torch

from zcls.config.key_word import KEY_OUTPUT
from zcls.model.recognizers.mnasnet_recognizer import MNASNetRecognizer, TorchvisionMNASNet


def test_mnasnet():
    data = torch.randn(1, 3, 224, 224)

    for wm in [0.5, 0.75, 1.0, 1.3]:
        model = MNASNetRecognizer(width_multiplier=wm)
        print(model)

        outputs = model(data)[KEY_OUTPUT]
        print(outputs.shape)

        assert outputs.shape == (1, 1000)

    for arch in ['mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3']:
        model = TorchvisionMNASNet(arch=arch)
        print(model)

        outputs = model(data)[KEY_OUTPUT]
        print(outputs.shape)

        assert outputs.shape == (1, 1000)


if __name__ == '__main__':
    test_mnasnet()

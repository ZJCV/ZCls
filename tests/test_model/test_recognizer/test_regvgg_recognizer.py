# -*- coding: utf-8 -*-

"""
@date: 2021/2/2 下午5:46
@file: test_regvgg_recognizer.py
@author: zj
@description: 
"""

import torch

from zcls.config.key_word import KEY_OUTPUT
from zcls.model.recognizers.repvgg_recognizer import RepVGGRecognizer
from zcls.model.recognizers.repvgg_recognizer import arch_settings


def test_regvgg_recognizer():
    data = torch.randn(1, 3, 224, 224)
    for key in arch_settings.keys():
        model = RepVGGRecognizer(arch=key)
        print(model)
        outputs = model(data)[KEY_OUTPUT]
        assert outputs.shape == (1, 1000)


if __name__ == '__main__':
    test_regvgg_recognizer()

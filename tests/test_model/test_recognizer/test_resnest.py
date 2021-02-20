# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午4:16
@file: test_resnest.py
@author: zj
@description: 
"""

import torch

from zcls.config.key_word import KEY_OUTPUT
from zcls.model.recognizers.resnet.official_resnest import OfficialResNeSt


def test_data(model, input_shape, output_shape):
    data = torch.randn(input_shape)
    outputs = model(data)[KEY_OUTPUT]
    print(outputs.shape)

    assert outputs.shape == output_shape


def test_resnest():
    # resnetd
    model = OfficialResNeSt(
        arch="resnest50_2s2x40d",
        num_classes=1000
    )
    print(model)
    test_data(model, (3, 3, 224, 224), (3, 1000))

    # resnetd
    model = OfficialResNeSt(
        arch="resnest50_2s2x40d_fast",
        num_classes=1000
    )
    print(model)
    test_data(model, (3, 3, 224, 224), (3, 1000))


if __name__ == '__main__':
    print('*' * 10 + ' resnetst')
    test_resnest()

# -*- coding: utf-8 -*-

"""
@date: 2020/12/29 上午9:41
@file: test_place_holder.py
@author: zj
@description: 
"""

import torch

from zcls.model.layers.place_holder import PlaceHolder


def test_place_holder():
    data = torch.randn(1, 3, 224, 224)
    model = PlaceHolder()
    outputs = model(data)

    print(outputs.shape)
    print(torch.allclose(data, outputs))


if __name__ == '__main__':
    test_place_holder()
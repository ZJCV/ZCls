# -*- coding: utf-8 -*-

"""
@date: 2020/12/30 下午6:34
@file: test_mnasnet_beckbone.py
@author: zj
@description: 
"""

import torch

from zcls.model.backbones.mobilenet.mnasnet_backbone import MNASNetBackbone
from zcls.model.backbones.misc import round_to_multiple_of


def test_mnasnet_backbone():
    data = torch.randn(1, 3, 224, 224)

    for wm in [0.5, 0.75, 1.0, 1.3]:
        model = MNASNetBackbone(width_multiplier=wm)
        # print(model)
        outputs = model(data)
        print(wm, outputs.shape)

        assert outputs.shape == (1, 1280, 7, 7)


def test():
    print(round_to_multiple_of(320 * 1.3, 8))


if __name__ == '__main__':
    test_mnasnet_backbone()
    test()

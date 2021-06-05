# -*- coding: utf-8 -*-

"""
@date: 2021/6/4 下午4:34
@file: test_ghost_backbone.py
@author: zj
@description: 
"""

import torch

from zcls.model.backbones.ghostnet.ghost_backbone import GhostBackbone, build_ghostnet_backbone
from zcls.model.backbones.misc import round_to_multiple_of


def test_backbone():
    data = torch.randn(2, 3, 224, 224)

    model = GhostBackbone()
    print(model)


if __name__ == '__main__':
    test_backbone()

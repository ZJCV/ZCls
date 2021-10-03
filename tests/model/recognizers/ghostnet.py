# -*- coding: utf-8 -*-

"""
@date: 2021/10/3 下午9:15
@file: shufflenet.py
@author: zj
@description: 
"""
import torch

from zcls.config import cfg
from zcls.model.recognizers.build import build_recognizer


def ghostnet():
    device = torch.device('cpu')

    cfg_file = 'tests/configs/ghostnet/ghostnet_x1_0.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)


if __name__ == '__main__':
    ghostnet()

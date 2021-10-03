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


def shufflenet():
    device = torch.device('cpu')

    cfg_file = 'tests/configs/shufflenet/shufflenet_v1_3g1x.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)

    cfg_file = 'tests/configs/shufflenet/shufflenet_v2_torchvision.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)

    cfg_file = 'tests/configs/shufflenet/shufflenet_v2_x2_0.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)


if __name__ == '__main__':
    shufflenet()

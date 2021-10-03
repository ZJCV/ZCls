# -*- coding: utf-8 -*-

"""
@date: 2021/10/3 下午3:48
@file: mobilenet.py
@author: zj
@description: 
"""

import torch

from zcls.config import cfg
from zcls.model.recognizers.build import build_recognizer


def mbv1():
    device = torch.device('cpu')

    cfg_file = 'tests/configs/mobilenet/mobilenet_v1_x1_0.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)


def mbv2():
    device = torch.device('cpu')

    cfg_file = 'tests/configs/mobilenet/mobilenet_v2_x1_0.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)

    cfg_file = 'tests/configs/mobilenet/mobilenet_v2_x1_0_torchvision.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)


def mbv3():
    device = torch.device('cpu')

    cfg_file = 'tests/configs/mobilenet/mobilenet_v3_large_x1_0.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)

    cfg_file = 'tests/configs/mobilenet/mobilenet_v3_large_se_x1_0.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)

    cfg_file = 'tests/configs/mobilenet/mobilenet_v3_large_se_hsigmoid_x1_0.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)

    cfg_file = 'tests/configs/mobilenet/mobilenet_v3_small_x1_0.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)

    cfg_file = 'tests/configs/mobilenet/mobilenet_v3_small_se_x1_0.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)

    cfg_file = 'tests/configs/mobilenet/mobilenet_v3_small_se_hsigmoid_x1_0.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)


def mnasnet():
    device = torch.device('cpu')

    cfg_file = 'tests/configs/mobilenet/mnasnet_a1_x1_3.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)

    cfg_file = 'tests/configs/mobilenet/mnasnet_a1_se_x1_3.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)

    cfg_file = 'tests/configs/mobilenet/mnasnet_b1_x1_3.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)

    cfg_file = 'tests/configs/mobilenet/mnasnet_b1_x1_3_torchvision.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)


if __name__ == '__main__':
    mbv1()
    mbv2()
    mbv3()
    mnasnet()

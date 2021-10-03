# -*- coding: utf-8 -*-

"""
@date: 2021/10/3 下午9:01
@file: resnet.py
@author: zj
@description: 
"""
import torch

from zcls.config import cfg
from zcls.model.recognizers.build import build_recognizer


def resnet():
    device = torch.device('cpu')

    cfg_file = 'tests/configs/resnet/resnet50.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)

    cfg_file = 'tests/configs/resnet/resnet50_torchvision.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)

    cfg_file = 'tests/configs/resnet/resnet3d50.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)


def resnext():
    device = torch.device('cpu')

    cfg_file = 'tests/configs/resnet/resnext50_32x4d.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)

    cfg_file = 'tests/configs/resnet/resnext50_32x4d_torchvision.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)

    cfg_file = 'tests/configs/resnet/resnextd50_32x4d_acb.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)

    cfg_file = 'tests/configs/resnet/resnextd50_32x4d_avg.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)

    cfg_file = 'tests/configs/resnet/resnextd50_32x4d_fast_avg.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)

    cfg_file = 'tests/configs/resnet/sknet50.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)


def resnest():
    device = torch.device('cpu')

    cfg_file = 'tests/configs/resnet/resnest50_2s2x40d.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)

    cfg_file = 'tests/configs/resnet/resnest50_2s2x40d_official.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)

    cfg_file = 'tests/configs/resnet/resnest50_fast_2s2x40d.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)

    cfg_file = 'tests/configs/resnet/resnest50_fast_2s2x40d_official.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)


if __name__ == '__main__':
    resnet()
    resnext()
    resnest()

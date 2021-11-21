# -*- coding: utf-8 -*-

"""
@date: 2021/10/3 下午9:01
@file: resnet-resnext.py
@author: zj
@description: 
"""
import torch

from zcls.config import cfg
from zcls.model.recognizers.build import build_recognizer


def resnet():
    device = torch.device('cpu')

    cfg_file = 'tests/configs/resnet-resnext/resnet-resnext/r18_zcls.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)

    cfg_file = 'tests/configs/resnet-resnext/resnet-resnext/r34_zcls.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)

    cfg_file = 'tests/configs/resnet-resnext/resnet-resnext/r50_zcls.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)

    cfg_file = 'tests/configs/resnet-resnext/resnet-resnext/r101_zcls.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)

    cfg_file = 'tests/configs/resnet-resnext/resnet-resnext/r152_zcls.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)

    cfg_file = 'tests/configs/resnet-resnext/resnet-resnext/r50_torchvision.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)


def resnet3d():
    device = torch.device('cpu')

    cfg_file = 'tests/configs/resnet-resnext/r3d50_zcls.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)


def resnext():
    device = torch.device('cpu')

    cfg_file = 'tests/configs/resnet-resnext/resnext/resnext50_32x4d_zcls.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)

    cfg_file = 'tests/configs/resnet-resnext/resnext/resnext50_32x4d_torchvision.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)

    cfg_file = 'tests/configs/resnet-resnext/resnext/resnextd50_32x4d_acb_zcls.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)

    cfg_file = 'tests/configs/resnet-resnext/resnext/resnextd50_32x4d_avg_zcls.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)

    cfg_file = 'tests/configs/resnet-resnext/resnext/resnextd50_32x4d_fast_avg_zcls.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)

    cfg_file = 'tests/configs/resnet-resnext/resnext/resnext101_32x8d_zcls.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)

    cfg_file = 'tests/configs/resnet-resnext/resnext/resnext101_32x8d_torchvision.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)


def sknet():
    device = torch.device('cpu')

    cfg_file = 'tests/configs/resnet-resnext/sknet50.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)


def resnest():
    device = torch.device('cpu')

    cfg_file = 'tests/configs/resnet-resnext/senet-skent-resnest/resnest50_2s2x40d.yaml'
    cfg.merge_from_file(cfg_file)

    model = build_recognizer(cfg, device)
    print(model)

    # cfg_file = 'tests/configs/resnet-resnext/senet-skent-resnest/resnest50_2s2x40d_official.yaml'
    # cfg.merge_from_file(cfg_file)
    #
    # model = build_recognizer(cfg, device)
    # print(model)
    #
    # cfg_file = 'tests/configs/resnet-resnext/senet-skent-resnest/resnest50_fast_2s2x40d.yaml'
    # cfg.merge_from_file(cfg_file)
    #
    # model = build_recognizer(cfg, device)
    # print(model)
    #
    # cfg_file = 'tests/configs/resnet-resnext/senet-skent-resnest/resnest50_fast_2s2x40d_official.yaml'
    # cfg.merge_from_file(cfg_file)
    #
    # model = build_recognizer(cfg, device)
    # print(model)


if __name__ == '__main__':
    # resnet-resnext()
    # resnet3d()
    # resnext()
    # sknet()
    resnest()

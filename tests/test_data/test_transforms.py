# -*- coding: utf-8 -*-

"""
@date: 2021/3/12 下午2:19
@file: test_transforms.py
@author: zj
@description: 
"""

import torch
import numpy as np
from PIL import Image

from zcls.config import cfg
from zcls.data.transforms.build import build_transform


def test_transforms():
    print(cfg.TRANSFORM)

    tran, _ = build_transform(cfg, is_train=True)
    print(tran)
    data = Image.fromarray(torch.randn(234, 134, 3).numpy().astype(np.uint8))
    print(tran(data).shape)

    tran, _ = build_transform(cfg, is_train=False)
    print(tran)
    data = Image.fromarray(torch.randn(231, 231, 3).numpy().astype(np.uint8))
    print(tran(data).shape)


def test_cfg():
    cfg_file = 'configs/cifar/rd50_cifar100_224_e100_sgd.yaml'
    cfg.merge_from_file(cfg_file)
    print(cfg.TRANSFORM)

    tran, _ = build_transform(cfg, is_train=True)
    data = Image.fromarray(torch.randn(234, 134, 3).numpy().astype(np.uint8))
    print(tran(data).shape)

    cfg_file = 'configs/cifar/rd50_cifar100_224_e100_sgd.yaml'
    cfg.merge_from_file(cfg_file)
    print(cfg.TRANSFORM)

    tran, _ = build_transform(cfg, is_train=True)
    data = Image.fromarray(torch.randn(234, 134, 3).numpy().astype(np.uint8))
    print(tran(data).shape)


def test_square_pad():
    config_file = 'tests/configs/square_pad.yaml'
    cfg.merge_from_file(config_file)
    print(cfg.TRANSFORM)

    tran, _ = build_transform(cfg, is_train=True)

    data = Image.fromarray(torch.randn(234, 134, 3).numpy().astype(np.uint8))
    res = tran(data)
    print(type(res))
    print(res.size)


if __name__ == '__main__':
    # test_transforms()
    # test_cfg()
    test_square_pad()

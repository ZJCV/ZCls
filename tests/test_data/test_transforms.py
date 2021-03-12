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

    res = build_transform(cfg, is_train=True)
    print(res)
    data = Image.fromarray(torch.randn(234, 134, 3).numpy().astype(np.uint8))
    print(res(data).shape)

    res = build_transform(cfg, is_train=False)
    print(res)
    data = Image.fromarray(torch.randn(231, 231, 3).numpy().astype(np.uint8))
    print(res(data).shape)


def test_cfg():
    cfg_file = 'configs/cifar/mbv3_large_se_hsigmoid_c100_224_e100_rmsprop_mslr_g1.yaml'
    cfg.merge_from_file(cfg_file)
    print(cfg.TRANSFORM)

    res = build_transform(cfg, is_train=True)
    data = Image.fromarray(torch.randn(234, 134, 3).numpy().astype(np.uint8))
    print(res(data).shape)

    cfg_file = 'configs/cifar/rxtd50_32x4d_c100_224_e100_sgd_mslr_g1.yaml'
    cfg.merge_from_file(cfg_file)
    print(cfg.TRANSFORM)

    res = build_transform(cfg, is_train=True)
    data = Image.fromarray(torch.randn(234, 134, 3).numpy().astype(np.uint8))
    print(res(data).shape)


if __name__ == '__main__':
    test_transforms()
    test_cfg()

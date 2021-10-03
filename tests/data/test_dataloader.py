# -*- coding: utf-8 -*-

"""
@date: 2021/4/6 下午4:11
@file: test_dataloader.py
@author: zj
@description: 
"""

from zcls.config import cfg
from zcls.data.build import build_data


def test_dataloader():
    config_file = 'configs/cifar/rd50_cifar100_224_e100_sgd.yaml'
    cfg.merge_from_file(config_file)

    dataloader = build_data(cfg, is_train=True)
    print(dataloader)
    te = iter(dataloader)
    print(te)

    images, targets = te.__next__()
    print(images.shape)
    print(targets)


if __name__ == '__main__':
    test_dataloader()

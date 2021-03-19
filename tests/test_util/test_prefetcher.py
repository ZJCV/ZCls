# -*- coding: utf-8 -*-

"""
@date: 2021/3/19 上午9:20
@file: test_prefetcher.py
@author: zj
@description: 
"""

import time
from tqdm import tqdm

from zcls.config import cfg
from zcls.util.prefetcher import Prefetcher
from zcls.data.build import build_dataloader


def test_prefetcher():
    config_file = 'configs/cifar/mbv3_large_se_hsigmoid_c10_224_e100_rmsprop_mslr_g1.yaml'
    cfg.merge_from_file(config_file)
    data_loader = build_dataloader(cfg, is_train=False)
    print(data_loader.__len__())

    print('enumerate')
    model = Prefetcher(data_loader)
    print(model)
    for i, item in enumerate(model):
        print(i)

    print('tqdm')
    model = Prefetcher(data_loader)
    print(model)
    for images, targets in tqdm(model):
        time.sleep(0.1)
    print('done')


if __name__ == '__main__':
    test_prefetcher()

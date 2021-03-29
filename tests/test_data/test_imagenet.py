# -*- coding: utf-8 -*-

"""
@date: 2021/3/28 下午7:11
@file: test_imagenet.py
@author: zj
@description: 
"""

import cv2
import numpy as np

from zcls.config import cfg
from zcls.data.datasets.imagenet import ImageNet
from zcls.data.build import build_dataloader


def test_dataloader():
    config_file = 'configs/imagenet/rxtd50_32x4d_imagenet_224_e100_sgd_mslr_e100_g2.yaml'
    cfg.merge_from_file(config_file)

    dataloader = build_dataloader(cfg, is_train=True)
    print(dataloader)
    te = iter(dataloader)
    print(te)

    images, targets = te.__next__()
    print(images.shape)
    print(targets)


def test_imagenet():
    data_root = 'data/imagenet'

    train_dataset = ImageNet(data_root, train=True)
    print(train_dataset)
    print('length:', len(train_dataset))

    img, target = train_dataset.__getitem__(30000)
    print(type(img), target)

    np_img = np.array(img)
    print(np_img.shape)
    cv2.imwrite('test_imagenet.png', np_img)


if __name__ == '__main__':
    # test_imagenet()
    test_dataloader()

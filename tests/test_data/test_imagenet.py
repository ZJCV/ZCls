# -*- coding: utf-8 -*-

"""
@date: 2021/3/28 下午7:11
@file: test_imagenet.py
@author: zj
@description: 
"""

import cv2
import numpy as np

from zcls.data.datasets.imagenet import ImageNet
from zcls.data.datasets.lmdb_imagenet import get_imagenet_classes, LMDBImageNet


def test_imagenet():
    data_root = 'data/imagenet'

    train_dataset = ImageNet(data_root, train=True)
    print(train_dataset)
    print('length:', len(train_dataset))
    print(train_dataset.classes)

    img, target = train_dataset.__getitem__(30000)
    print(type(img), target)

    np_img = np.array(img)
    print(np_img.shape)
    cv2.imwrite('test_imagenet.png', np_img)


def test_imagenet_classes():
    idx2label = get_imagenet_classes()
    print(idx2label)


def test_lmdb_imagenet():
    data_root = 'data/imagenet/train.lmdb'

    train_dataset = LMDBImageNet(data_root)
    print(train_dataset)
    print('length:', len(train_dataset))
    print('classes:', train_dataset.classes)

    img, target = train_dataset.__getitem__(30000)
    print(type(img), target)

    np_img = np.array(img)
    print(np_img.shape)
    cv2.imwrite('test_imagenet.png', np_img)


if __name__ == '__main__':
    test_imagenet()
    test_imagenet_classes()
    test_lmdb_imagenet()

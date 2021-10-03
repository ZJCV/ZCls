# -*- coding: utf-8 -*-

"""
@date: 2021/10/3 下午3:13
@file: build.py
@author: zj
@description: 
"""
import cv2

from zcls.config import cfg
from zcls.data.transforms.build import build_transform


def transform():
    image = cv2.imread('tests/assets/lena_224.jpg')
    print('image.shape:', image.shape)

    cfg_file = 'tests/configs/transforms.yaml'
    cfg.merge_from_file(cfg_file)

    t_train, _ = build_transform(cfg, is_train=True)
    print(t_train)

    res = t_train(image)
    print('res.shape:', res.shape)
    cv2.imwrite('tests/assets/lena_t_train.jpg', res)

    t_test, _ = build_transform(cfg, is_train=True)
    print(t_test)

    res = t_test(image)
    print('res.shape:', res.shape)
    cv2.imwrite('tests/assets/lena_t_test.jpg', res)


if __name__ == '__main__':
    transform()

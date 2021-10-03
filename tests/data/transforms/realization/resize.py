# -*- coding: utf-8 -*-

"""
@date: 2021/10/3 上午11:24
@file: resize.py
@author: zj
@description: 
"""

import cv2

from zcls.data.transforms.realization import Resize


def resize():
    image = cv2.imread('tests/assets/lena_224.jpg')
    print('image.shape:', image.shape)

    # enlarge
    m = Resize(size=(256, 256), interpolation=cv2.INTER_LINEAR, p=1.0)
    print(m)
    res = m(image)
    print('enlarged image.shape:', res.shape)
    # cv2.imwrite('tests/assets/lena_256.jpg', res)

    # reduce
    m = Resize(size=(112, 112), interpolation=cv2.INTER_AREA, p=1.0)
    print(m)
    res = m(image)
    print('reduced image.shape:', res.shape)
    # cv2.imwrite('tests/assets/lena_112.jpg', res)


if __name__ == '__main__':
    resize()

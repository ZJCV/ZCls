# -*- coding: utf-8 -*-

"""
@date: 2021/10/3 上午11:09
@file: rotate.py
@author: zj
@description: 
"""

import cv2

from zcls.data.transforms.realization import Rotate


def rotate():
    image = cv2.imread('tests/assets/lena_224.jpg')

    m = Rotate((-30, 30), interpolation=cv2.INTER_LINEAR,
               border_mode=cv2.BORDER_REFLECT_101, value=None, p=0.5)
    print(m)
    res = m(image)
    cv2.imwrite('tests/assets/lena_ro.jpg', res)

    m = Rotate((-30, 30), interpolation=cv2.INTER_LINEAR,
               border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5)
    print(m)
    res = m(image)
    cv2.imwrite('tests/assets/lena_ro2.jpg', res)


if __name__ == '__main__':
    rotate()

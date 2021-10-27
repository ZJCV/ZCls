# -*- coding: utf-8 -*-

"""
@date: 2021/10/3 下午3:06
@file: dropout.py
@author: zj
@description: 
"""

import cv2

from zcls.data.transforms.realization import ColorJitter


def brightness():
    m = ColorJitter(brightness=0.2, contrast=0., saturation=0., hue=0., p=1.0)
    print(m)

    image = cv2.imread('tests/assets/lena_224.jpg')
    res = m(image)
    cv2.imwrite('tests/assets/lena_brightness.jpg', res)


def contrast():
    m = ColorJitter(brightness=0., contrast=0.2, saturation=0., hue=0., p=1.0)
    print(m)

    image = cv2.imread('tests/assets/lena_224.jpg')
    res = m(image)
    cv2.imwrite('tests/assets/lena_contrast.jpg', res)


def saturation():
    m = ColorJitter(brightness=0., contrast=0., saturation=0.2, hue=0., p=1.0)
    print(m)

    image = cv2.imread('tests/assets/lena_224.jpg')
    res = m(image)
    cv2.imwrite('tests/assets/lena_saturation.jpg', res)


def hue():
    m = ColorJitter(brightness=0., contrast=0., saturation=0., hue=0.2, p=1.0)
    print(m)

    image = cv2.imread('tests/assets/lena_224.jpg')
    res = m(image)
    cv2.imwrite('tests/assets/lena_hue.jpg', res)


def all():
    image = cv2.imread('tests/assets/lena_224.jpg')

    m = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=1.0)
    print(m)

    res = m(image)
    cv2.imwrite('tests/assets/lena_color_jitter.jpg', res)


if __name__ == '__main__':
    brightness()
    contrast()
    saturation()
    hue()
    all()

# -*- coding: utf-8 -*-

"""
@date: 2021/10/4 上午9:48
@file: augment.py
@author: zj
@description: 
"""

import cv2

from zcls.data.transforms.realization.autoaugment import AutoAugment
from torchvision.transforms.autoaugment import AutoAugmentPolicy


def aug():
    image = cv2.imread('tests/assets/lena_224.jpg')

    m = AutoAugment(policy=AutoAugmentPolicy.IMAGENET, p=1.0)
    print(m)
    res = m(image)
    cv2.imwrite('tests/assets/lena_aug.jpg', res)


if __name__ == '__main__':
    aug()

# -*- coding: utf-8 -*-

"""
@date: 2021/10/3 下午3:08
@file: flip.py
@author: zj
@description:
"""

import numpy as np

from zcls.data.transforms.realization import HorizontalFlip, VerticalFlip


def horizontal_flip():
    data = np.arange(3 * 5).reshape(3, 5)
    print(data)
    # [[ 0  1  2  3  4]
    #  [ 5  6  7  8  9]
    #  [10 11 12 13 14]]

    m = HorizontalFlip(p=1.0)
    print(m)

    res = m(data)
    print(res)
    # [[ 4  3  2  1  0]
    #  [ 9  8  7  6  5]
    #  [14 13 12 11 10]]


def vertical_flip():
    data = np.arange(3 * 5).reshape(3, 5)
    print(data)
    # [[ 0  1  2  3  4]
    #  [ 5  6  7  8  9]
    #  [10 11 12 13 14]]

    m = VerticalFlip(p=1.0)
    print(m)

    res = m(data)
    print(res)
    # [[10 11 12 13 14]
    #  [ 5  6  7  8  9]
    #  [ 0  1  2  3  4]]


if __name__ == '__main__':
    horizontal_flip()
    vertical_flip()

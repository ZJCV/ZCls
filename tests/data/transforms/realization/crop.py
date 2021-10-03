# -*- coding: utf-8 -*-

"""
@date: 2021/10/3 上午11:30
@file: random_crop.py
@author: zj
@description: 
"""

import numpy as np

from zcls.data.transforms.realization import CenterCrop, RandomCrop


def center_crop():
    data = np.arange(3 * 5).reshape(3, 5)
    print(data)
    # [[ 0  1  2  3  4]
    #  [ 5  6  7  8  9]
    #  [10 11 12 13 14]]

    m = CenterCrop(size=(3, 3), p=1.0)
    print(m)

    res = m(data)
    print(res)
    # [[ 1  2  3]
    #  [ 6  7  8]
    #  [11 12 13]]


def random_crop():
    data = np.arange(3 * 5).reshape(3, 5)
    print(data)
    # [[ 0  1  2  3  4]
    #  [ 5  6  7  8  9]
    #  [10 11 12 13 14]]

    m = RandomCrop(size=(3, 3), p=1.0)
    print(m)

    res = m(data)
    print(res)


if __name__ == '__main__':
    center_crop()
    random_crop()

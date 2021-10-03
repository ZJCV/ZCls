# -*- coding: utf-8 -*-

"""
@date: 2021/10/3 上午10:55
@file: pad.py
@author: zj
@description: 
"""

import numpy as np

from zcls.data.transforms.realization.square_pad import SquarePad


def square_pad():
    m = SquarePad()

    a = np.ones((3, 8))
    print(a)
    # [[1. 1. 1. 1. 1. 1. 1. 1.]
    #  [1. 1. 1. 1. 1. 1. 1. 1.]
    #  [1. 1. 1. 1. 1. 1. 1. 1.]]

    res = m(a)
    print(res)
    # [[0. 0. 0. 0. 0. 0. 0. 0.]
    #  [0. 0. 0. 0. 0. 0. 0. 0.]
    #  [1. 1. 1. 1. 1. 1. 1. 1.]
    #  [1. 1. 1. 1. 1. 1. 1. 1.]
    #  [1. 1. 1. 1. 1. 1. 1. 1.]
    #  [0. 0. 0. 0. 0. 0. 0. 0.]
    #  [0. 0. 0. 0. 0. 0. 0. 0.]
    #  [0. 0. 0. 0. 0. 0. 0. 0.]]


if __name__ == '__main__':
    square_pad()

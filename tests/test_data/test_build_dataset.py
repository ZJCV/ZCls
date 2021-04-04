# -*- coding: utf-8 -*-

"""
@date: 2021/3/31 下午4:53
@file: test_build_dataset.py
@author: zj
@description: 
"""

from zcls.data.datasets.build import build_dataset
from zcls.data.datasets.cifar import CIFAR


def test_build_dataset():
    model = CIFAR
    print(type(model))
    print(isinstance(model, CIFAR))


if __name__ == '__main__':
    test_build_dataset()

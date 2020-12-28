# -*- coding: utf-8 -*-

"""
@date: 2020/12/3 下午8:27
@file: test_mobilenetv1_recognizer.py
@author: zj
@description: 
"""

import torch
from torchvision.models import mnasnet


def test_mnasnet():
    model = mnasnet.mnasnet1_0()
    print(model)


if __name__ == '__main__':
    test_mnasnet()

# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午1:54
@file: build.py
@author: zj
@description: 
"""

import torch.nn as nn


def build_criterion():
    return nn.CrossEntropyLoss()

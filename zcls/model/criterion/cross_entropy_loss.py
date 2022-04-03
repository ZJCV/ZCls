# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午5:07
@file: cross_entropy_loss.py
@author: zj
@description: 
"""

import torch.nn as nn


def build_cross_entropy_loss():
    return nn.CrossEntropyLoss()

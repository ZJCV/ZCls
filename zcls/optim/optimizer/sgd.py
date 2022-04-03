# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午5:00
@file: sgd.py
@author: zj
@description: 
"""
import torch
import torch.nn as nn


def build_sgd(model, lr=1e-3, momentum=0.9, weight_decay=1e-5):
    assert isinstance(model, nn.Module)
    return torch.optim.SGD(model.parameters(), lr,
                           momentum=momentum,
                           weight_decay=weight_decay)

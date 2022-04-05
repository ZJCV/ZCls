# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午5:00
@file: sgd.py
@author: zj
@description: 
"""
import torch


def build_sgd(groups, lr=1e-3, momentum=0.9, weight_decay=1e-5):
    return torch.optim.SGD(groups,
                           lr=lr,
                           momentum=momentum,
                           weight_decay=weight_decay)

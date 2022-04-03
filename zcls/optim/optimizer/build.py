# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午1:52
@file: build.py
@author: zj
@description: 
"""

import torch


def build_optimizer(args, model):
    return torch.optim.SGD(model.parameters(), args.lr,
                           momentum=args.momentum,
                           weight_decay=args.weight_decay)

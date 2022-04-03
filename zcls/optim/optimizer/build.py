# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午1:52
@file: build.py
@author: zj
@description: 
"""

from .sgd import build_sgd


def build_optimizer(args, model):
    if args.optimizer == 'sgd':
        return build_sgd(model,
                         lr=args.lr,
                         momentum=args.momentum,
                         weight_decay=args.weight_decay)
    else:
        raise ValueError(f"{args.optimizer} doesn't support")

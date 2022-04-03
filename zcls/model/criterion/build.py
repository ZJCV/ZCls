# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午1:54
@file: build.py
@author: zj
@description: 
"""

from .cross_entropy_loss import build_cross_entropy_loss


def build_criterion(args):
    if args.loss == 'CrossEntropyLoss':
        return build_cross_entropy_loss()
    else:
        raise ValueError(f"{args.loss} doesn't support")

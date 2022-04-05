# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午1:54
@file: build.py
@author: zj
@description: 
"""

from .cross_entropy_loss import build_cross_entropy_loss
from .large_margin_softmax_loss import build_large_margin_softmax_loss


def build_criterion(args):
    if args.loss == 'CrossEntropyLoss':
        return build_cross_entropy_loss()
    elif args.loss == 'LargeMarginSoftmaxV1':
        return build_large_margin_softmax_loss(args)
    else:
        raise ValueError(f"{args.loss} doesn't support")

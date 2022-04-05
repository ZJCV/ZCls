# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午1:52
@file: build.py
@author: zj
@description: 
"""

import torch.nn as nn

from .sgd import build_sgd


def build_optimizer(args, model):
    assert isinstance(model, nn.Module)
    groups = filter_weight(model)

    if args.optimizer == 'sgd':
        return build_sgd(groups,
                         lr=args.lr,
                         momentum=args.momentum,
                         weight_decay=args.weight_decay)
    else:
        raise ValueError(f"{args.optimizer} doesn't support")


def filter_weight(module):
    """
    1. Avoid bias of all layers and normalization layer for weight decay.
    2. And filter all layers which require_grad=False
    refer to
    1. [Allow to set 0 weight decay for biases and params in batch norm #1402](https://github.com/pytorch/pytorch/issues/1402)
    2. [Weight decay in the optimizers is a bad idea (especially with BatchNorm)](https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994)
    """
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                # NO Linear BIAS
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                # NO Conv BIAS
                group_no_decay.append(m.bias)
        elif isinstance(m, (nn.modules.batchnorm._BatchNorm, nn.GroupNorm, nn.LayerNorm)):
            # NO BN Weights / BIAS
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)

    new_group_decay = filter(lambda p: p.requires_grad, group_decay)
    new_group_no_decay = filter(lambda p: p.requires_grad, group_no_decay)
    groups = [dict(params=new_group_decay), dict(params=new_group_no_decay, weight_decay=0.)]
    return groups

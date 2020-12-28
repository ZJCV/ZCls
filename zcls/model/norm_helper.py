# -*- coding: utf-8 -*-

"""
@date: 2020/9/23 下午2:35
@file: norm_helper.py
@author: zj
@description: 
"""

import torch.nn as nn
from functools import partial

from .layers.group_norm_wrapper import GroupNormWrapper


def convert_sync_bn(model, process_group):
    sync_bn_module = nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)
    return sync_bn_module


def get_norm(cfg):
    """
    Args:
        cfg (CfgNode): model building configs, details are in the comments of
            the config file.
    Returns:
        nn.Module: the normalization layer.
    """
    norm_type = cfg.MODEL.NORM.TYPE
    if norm_type == "BatchNorm2d":
        return nn.BatchNorm2d
    elif norm_type == 'BatchNorm3d':
        return nn.BatchNorm3d
    elif norm_type == "GroupNorm":
        num_groups = cfg.MODEL.NORM.GROUPS
        return partial(GroupNormWrapper, num_groups=num_groups)
    else:
        raise NotImplementedError(
            "Norm type {} is not supported".format(norm_type)
        )


def freezing_bn(model, partial_bn=False):
    count = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            count += 1
            if count == 1 and partial_bn:
                continue

            m.eval()
            # shutdown update in frozen mode
            m.weight.requires_grad = False
            m.bias.requires_grad = False

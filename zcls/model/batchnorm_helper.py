# -*- coding: utf-8 -*-

"""
@date: 2020/9/23 下午2:35
@file: batchnorm_helper.py
@author: zj
@description: 
"""

import torch.nn as nn
from functools import partial

from .layers.group_norm_wrapper import GroupNormWrapper


def convert_sync_bn(model, process_group, device):
    # convert all BN layers in the model to syncBN
    for _, (child_name, child) in enumerate(model.named_children()):
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            m = nn.SyncBatchNorm.convert_sync_batchnorm(child, process_group)
            m = m.to(device=device)
            setattr(model, child_name, m)
        else:
            convert_sync_bn(child, process_group, device)


def get_norm(cfg):
    """
    Args:
        cfg (CfgNode): model building configs, details are in the comments of
            the config file.
    Returns:
        nn.Module: the normalization layer.
    """
    if cfg.MODEL.NORM.TYPE == "BatchNorm2d":
        return nn.BatchNorm2d
    elif cfg.MODEL.NORM.TYPE == "GroupNorm":
        num_groups = cfg.MODEL.NORM.GROUPS
        return partial(GroupNormWrapper, num_groups=num_groups)
    else:
        raise NotImplementedError(
            "Norm type {} is not supported".format(cfg.BN.NORM_TYPE)
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

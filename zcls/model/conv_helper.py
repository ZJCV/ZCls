# -*- coding: utf-8 -*-

"""
@date: 2020/12/4 下午4:11
@file: act_helper.py
@author: zj
@description: 
"""

import torch.nn as nn


def get_conv(cfg):
    """
    Args:
        cfg (CfgNode): model building configs, details are in the comments of
            the config file.
    Returns:
        nn.Module: the conv layer.
    """
    conv_type = cfg.MODEL.CONV.TYPE
    if conv_type == "Conv2d":
        return nn.Conv2d
    elif conv_type == "Conv3d":
        return nn.Conv3d
    else:
        raise NotImplementedError(
            "Conv type {} is not supported".format(conv_type)
        )

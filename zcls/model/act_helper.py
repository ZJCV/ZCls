# -*- coding: utf-8 -*-

"""
@date: 2020/12/4 下午4:11
@file: act_helper.py
@author: zj
@description: 
"""

import torch.nn as nn

from .layers.hard_swish_wrapper import HardswishWrapper


def get_act(cfg):
    """
    Args:
        cfg (CfgNode): model building configs, details are in the comments of
            the config file.
    Returns:
        nn.Module: the activation layer.
    """
    act_type = cfg.MODEL.ACT.TYPE
    if act_type == "ReLU":
        return nn.ReLU
    elif act_type == "ReLU6":
        return nn.ReLU6
    elif act_type == 'HSwish':
        return HardswishWrapper
    else:
        raise NotImplementedError(
            "Activation type {} is not supported".format(act_type)
        )

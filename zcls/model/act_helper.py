# -*- coding: utf-8 -*-

"""
@date: 2020/12/4 下午4:11
@file: act_helper.py
@author: zj
@description: 
"""

import torch.nn as nn


def get_act(cfg):
    """
    Args:
        cfg (CfgNode): model building configs, details are in the comments of
            the config file.
    Returns:
        nn.Module: the normalization layer.
    """
    if cfg.MODEL.ACT.TYPE == "ReLU":
        return nn.ReLU
    elif cfg.MODEL.ACT.TYPE == "ReLU6":
        return nn.ReLU6
    else:
        raise NotImplementedError(
            "Norm type {} is not supported".format(cfg.MODEL.ACT.TYPE)
        )

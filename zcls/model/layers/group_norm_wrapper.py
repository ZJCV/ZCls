# -*- coding: utf-8 -*-

"""
@date: 2020/11/27 下午3:39
@file: group_norm_wrapper.py
@author: zj
@description: 
"""

import torch.nn as nn


class GroupNormWrapper(nn.GroupNorm):

    def __init__(self, num_channels: int, num_groups: int = 4, eps: float = 1e-5, affine: bool = True) -> None:
        super(GroupNormWrapper, self).__init__(num_groups, num_channels, eps, affine)

        self.num_features = num_channels

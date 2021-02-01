# -*- coding: utf-8 -*-

"""
@date: 2020/12/28 上午9:50
@file: place_holder.py
@author: zj
@description: 占位层，不执行任何操作，直接返回输入数据
"""
from abc import ABC

import torch.nn as nn


class PlaceHolder(nn.Module, ABC):
    """
    @deprecated. pytorch已实现，使用nn.Identity()
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

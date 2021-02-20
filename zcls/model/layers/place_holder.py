# -*- coding: utf-8 -*-

"""
@date: 2020/12/28 上午9:50
@file: place_holder.py
@author: zj
@description: The placeholder layer returns the input data directly without performing any operation
"""
from abc import ABC

import torch.nn as nn


class PlaceHolder(nn.Module, ABC):
    """
    @deprecated. pytorch is implemented, using nn.Identity()
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

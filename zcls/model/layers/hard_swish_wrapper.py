# -*- coding: utf-8 -*-

"""
@date: 2020/12/30 下午9:38
@file: hard_swish_wrapper.py
@author: zj
@description: 
"""
from abc import ABC

import torch.nn as nn


class HardswishWrapper(nn.Hardswish, ABC):

    def __init__(self, inplace: bool = False):
        super(HardswishWrapper, self).__init__()

# -*- coding: utf-8 -*-

"""
@date: 2020/11/26 下午10:47
@file: init_helper.py
@author: zj
@description: 
"""

import math
from torch.nn import init


def reset_parameters(layer) -> None:
    init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
    if layer.bias is not None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(layer.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(layer.bias, -bound, bound)

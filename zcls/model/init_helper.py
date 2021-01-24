# -*- coding: utf-8 -*-

"""
@date: 2020/11/26 下午10:47
@file: init_helper.py
@author: zj
@description: 
"""

import math
import torch.nn as nn
from torch.nn import init


def reset_parameters(layer) -> None:
    init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
    if layer.bias is not None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(layer.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(layer.bias, -bound, bound)


def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

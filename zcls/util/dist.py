# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午1:39
@file: dist.py
@author: zj
@description: 
"""

import torch.distributed as dist


def reduce_tensor(world_size, tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt

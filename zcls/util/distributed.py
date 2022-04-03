# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午1:39
@file: distributed.py
@author: zj
@description: 
"""

import torch
import torch.distributed as dist


def reduce_tensor(world_size, tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt


def is_master_proc():
    """
    Determines if the current process is the master process.
    """
    if torch.distributed.is_initialized():
        return dist.get_rank() % get_world_size() == 0
    else:
        return True


def get_world_size():
    """
    Get the size of the world.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

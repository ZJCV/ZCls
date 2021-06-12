# -*- coding: utf-8 -*-

"""
@date: 2021/3/18 下午2:13
@file: prefetcher.py
@author: zj
@description: 
"""

import torch
from torch.utils.data import DataLoader

from . import logging

logger = logging.get_logger(__name__)


class Prefetcher():
    """
    from [data_prefetcher](https://github.com/NVIDIA/apex/blob/f5cd5ae937f168c763985f627bbf850648ea5f3f/examples/imagenet/main_amp.py#L256)
    refert to:
    1. [Dose data_prefetcher() really speed up training? #304](https://github.com/NVIDIA/apex/issues/304)
    2. [如何给你PyTorch里的Dataloader打鸡血](https://zhuanlan.zhihu.com/p/66145913)
    Note:
    For overlapped prefetching, supplying pin_memory=True to the dataloader is always required
    """

    def __init__(self, loader: DataLoader, device):
        assert isinstance(loader, DataLoader)
        self.device = device
        self.length = len(loader)
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()

        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration as e:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            # self.next_input = self.next_input.cuda(non_blocking=True)
            # self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.to(self.device, non_blocking=True)
            self.next_target = self.next_target.to(self.device, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        if self.next_input is None:
            raise StopIteration
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target

    def __iter__(self):
        return self

    def __len__(self):
        return self.length

    def release(self):
        del self.stream
        del self.loader
        self.device = None
        self.length = None

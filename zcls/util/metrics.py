# -*- coding: utf-8 -*-

"""
@date: 2020/4/27 下午8:25
@file: metrics.py
@author: zj
@description: 
"""

from thop import profile
from thop import clever_format


def compute_num_flops(model, data):
    macs, params = profile(model, inputs=(data,), verbose=False)
    # print(macs, params)

    GFlops = macs * 2.0 / pow(10, 9)
    # 假定每个参数使用32位浮点数保存，那么需要4个字节
    params_size = params * 4.0 / 1024 / 1024
    return GFlops, params_size


def topk_accuracy(output, target, top_k=(1,)):
    """
    计算前K个。N表示样本数，C表示类别数
    :param output: 大小为[N, C]，每行表示该样本计算得到的C个类别概率
    :param target: 大小为[N]，每行表示指定类别
    :param topk: tuple，计算前top-k的accuracy
    :return: list
    """
    assert len(output.shape) == 2 and output.shape[0] == target.shape[0]
    maxk = max(top_k)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in top_k:
        # correct_k = correct[:k].view(-1).float().sum(0)
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# -*- coding: utf-8 -*-

"""
@date: 2021/2/19 下午2:30
@file: misc.py
@author: zj
@description: 
"""

import torch


def make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def round_to_multiple_of(val, divisor, round_up_bias=0.9):
    """ Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88. """
    assert 0.0 < round_up_bias < 1.0
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor


def channel_shuffle(x, groups):
    """
    # >>> a = torch.arange(12)
    # >>> b = a.reshape(3,4)
    # >>> c = b.transpose(1,0).contiguous()
    # >>> d = c.view(3,4)
    # >>> a
    # tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
    # >>> b
    # tensor([[ 0,  1,  2,  3],
    #         [ 4,  5,  6,  7],
    #         [ 8,  9, 10, 11]])
    # >>> c
    # tensor([[ 0,  4,  8],
    #         [ 1,  5,  9],
    #         [ 2,  6, 10],
    #         [ 3,  7, 11]])
    # >>> d
    # tensor([[ 0,  4,  8,  1],
    #         [ 5,  9,  2,  6],
    #         [10,  3,  7, 11]])
    """
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

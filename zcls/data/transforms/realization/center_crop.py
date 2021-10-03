# -*- coding: utf-8 -*-

"""
@date: 2021/10/3 上午11:27
@file: random_crop.py
@author: zj
@description: 
"""

import albumentations as A
from typing import Sequence


class CenterCrop(object):
    """
    Crop the central part of the input.

    Args:
        size (sequence): Desired output size.
        p (float): probability of applying the transform. Default: 1.

    Image types:
        uint8, float32

    Note:
        It is recommended to use uint8 images as input.
        Otherwise the operation will require internal conversion
        float32 -> uint8 -> float32 that causes worse performance.
    """

    def __init__(self, size, p=1.0):
        if not isinstance(size, Sequence):
            raise TypeError("Size should be int or sequence. Got {}".format(type(size)))
        if isinstance(size, Sequence) and len(size) not in (2,):
            raise ValueError("If size is a sequence, it should have 2 values")
        self.size = size
        self.p = p

        new_h, new_w = self.size
        self.t = A.CenterCrop(width=new_w, height=new_h, p=1.0)

    def __call__(self, image):
        return self.t(image=image)['image']

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, p={1})'. \
            format(self.size, self.p)

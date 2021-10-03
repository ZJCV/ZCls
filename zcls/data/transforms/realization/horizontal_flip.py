# -*- coding: utf-8 -*-

"""
@date: 2021/10/3 下午12:03
@file: horizontal_flip.py
@author: zj
@description: 
"""

import albumentations as A


class HorizontalFlip(object):
    """
    Flip the input horizontally around the y-axis.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Image types:
        uint8, float32
    """

    def __init__(self, p=0.5):
        self.p = p

        self.t = A.HorizontalFlip(p=self.p)

    def __call__(self, image):
        return self.t(image=image)['image']

    def __repr__(self):
        return self.__class__.__name__ + '(p={0})'.format(self.p)

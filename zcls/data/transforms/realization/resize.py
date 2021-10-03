# -*- coding: utf-8 -*-

"""
@date: 2021/10/3 上午11:17
@file: resize.py
@author: zj
@description: 
"""

import cv2

import albumentations as A
from typing import Sequence


class Resize(object):
    """
    Resize the input to the given size.
    when enlarged image, recommended cv2.INTER_LINEAR; when reduce image, recommended cv2.INTER_ARER.

    Args:
        size (sequence): Desired output size.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Image types:
        uint8, float32

    Related:
        torchvision PIL transform always use antialiasing so that's different comparing to other library. See
        https://github.com/pytorch/vision/blob/fbd69f1052292336f78b69804a68ade5a6b2f3b4/torchvision/transforms/functional.py#L357
    """

    def __init__(self, size, interpolation=cv2.INTER_LINEAR, p=1.0):
        if not isinstance(size, Sequence):
            raise TypeError("Size should be int or sequence. Got {}".format(type(size)))
        if isinstance(size, Sequence) and len(size) not in (2, ):
            raise ValueError("If size is a sequence, it should have 2 values")
        self.size = size

        assert interpolation in [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA,
                                 cv2.INTER_LANCZOS4]
        self.interpolation = interpolation
        self.p = p

        new_h, new_w = self.size
        self.t = A.Resize(new_h, new_w, interpolation=self.interpolation, p=self.p)

    def __call__(self, image):
        return self.t(image=image)['image']

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, interpolation={1}, p={2})'. \
            format(self.size, self.interpolation, self.p)

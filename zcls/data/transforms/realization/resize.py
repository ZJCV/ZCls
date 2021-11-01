# -*- coding: utf-8 -*-

"""
@date: 2021/10/3 上午11:17
@file: resize.py
@author: zj
@description: 
"""

import cv2
import numpy as np

from albumentations.augmentations.geometric import functional as F
from typing import Sequence


def get_hw(img: np.ndarray, size: int, mode: int):
    assert mode in [0, 1]
    h, w = img.shape[:2]

    short, long = (w, h) if w <= h else (h, w)
    if mode == 0:
        new_short, new_long = size, int(size * long / short)
    else:
        new_short, new_long = int(size * short / long), size

    new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
    return new_h, new_w


class Resize(object):
    """
    Resize the input image to the given size.
    when enlarged image, recommended cv2.INTER_LINEAR; when reduce image, recommended cv2.INTER_ARER.
    because torchvision PIL transform always use antialiasing so that's different comparing to other library. See
    https://github.com/pytorch/vision/blob/fbd69f1052292336f78b69804a68ade5a6b2f3b4/torchvision/transforms/functional.py#L357

    Latest: use albumentations' resize function

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size).
        mode (int): zoom to the largest(1) or smallest edge(0). Default: 0
        p (float): probability of applying the transform. Default: 1.
    """

    def __init__(self, size, interpolation=cv2.INTER_LINEAR, mode=0, p=1.0):
        if p != 1.0:
            raise ValueError('p should always be 1.0')
        if not isinstance(size, (int, Sequence)):
            raise TypeError("Size should be int or sequence. Got {}".format(type(size)))
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")

        self.size = size

        assert interpolation in [cv2.INTER_AREA, cv2.INTER_LINEAR]
        self.interpolation = interpolation

        assert mode in [0, 1]
        self.mode = 2 if len(self.size) == 2 else mode

    def __call__(self, img):
        if self.mode == 2:
            new_h, new_w = self.size
        else:
            new_h, new_w = get_hw(img, list(self.size)[0], self.mode)

        new_img = F.resize(img, new_h, new_w, interpolation=self.interpolation)
        return new_img

    def __repr__(self):
        interpolate_str = 'INTER_AREA' if self.interpolation == cv2.INTER_AREA else 'INTER_LINEAR'
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)

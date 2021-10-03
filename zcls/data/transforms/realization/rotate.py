# -*- coding: utf-8 -*-

"""
@date: 2021/7/23 下午9:27
@file: square_pad.py
@author: zj
@description: 
"""

import cv2
import albumentations as A


class Rotate(object):
    """
    Rotate the input by an angle selected randomly from the uniform distribution.

    Args:
        limit ((int, int) or int): range from which a random angle is picked. If limit is a single int
            an angle is picked from (-limit, limit). Default: (-90, 90)
        border_mode (OpenCV flag): flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
            Default: cv2.BORDER_REFLECT_101
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        value (int, float, list of ints, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
        p (float): probability of applying the transform. Default: 0.5.
    """

    def __init__(self, limit, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, value=None,
                 p=0.5):
        self.limit = limit
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.p = p
        self.t = A.Rotate(limit=self.limit, interpolation=self.interpolation,
                          border_mode=self.border_mode, value=self.value, p=self.p)

    def __call__(self, image):
        return self.t(image=image)['image']

    def __repr__(self):
        return self.__class__.__name__ + '(limit={0}, interpolation={1}, border_mode={2}, value={3}, p={4})'. \
            format(self.limit, self.interpolation, self.border_mode, self.value, self.p)

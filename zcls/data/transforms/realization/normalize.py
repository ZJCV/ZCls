# -*- coding: utf-8 -*-

"""
@date: 2021/10/3 下午12:06
@file: color_jitter.py
@author: zj
@description: 
"""

import albumentations as A


class Normalize(object):
    """
    Normalization is applied by the formula: `img = (img - mean * max_pixel_value) / (std * max_pixel_value)`

    Args:
        mean (float, list of float): mean values
        std  (float, list of float): std values
        max_pixel_value (float): maximum possible pixel value

    Image types:
        uint8, float32
    """

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0):
        self.mean = mean
        self.std = std
        self.max_pixel_value = max_pixel_value
        self.p = p

        self.t = A.Normalize(mean=self.mean, std=self.std, max_pixel_value=self.max_pixel_value, p=self.p)

    def __call__(self, image):
        return self.t(image=image)['image']

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1}, max_pixel_value={2}, p={3})' \
            .format(self.mean, self.std, self.max_pixel_value, self.p)

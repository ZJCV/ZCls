# -*- coding: utf-8 -*-

"""
@date: 2021/10/3 下午12:06
@file: color_jitter.py
@author: zj
@description: 
"""

import albumentations as A


class CoarseDropout(object):
    """
    CoarseDropout of the rectangular regions in the image.

    Args:
        max_holes (int): Maximum number of regions to zero out.
        max_height (int): Maximum height of the hole.
        max_width (int): Maximum width of the hole.
        min_holes (int): Minimum number of regions to zero out. If `None`,
            `min_holes` is be set to `max_holes`. Default: `None`.
        min_height (int): Minimum height of the hole. Default: None. If `None`,
            `min_height` is set to `max_height`. Default: `None`.
        min_width (int): Minimum width of the hole. If `None`, `min_height` is
            set to `max_width`. Default: `None`.
        fill_value (int, float, list of int, list of float): value for dropped pixels.

    Image types:
        uint8, float32

    Reference:
    |  https://arxiv.org/abs/1708.04552
    |  https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
    |  https://github.com/aleju/imgaug/blob/master/imgaug/augmenters/arithmetic.py
    """

    def __init__(self, max_holes=8, max_height=8, max_width=8, min_holes=None, min_height=None, min_width=None,
                 fill_value=0, p=0.5):
        self.max_holes = max_holes
        self.max_height = max_height
        self.max_width = max_width
        self.min_holes = min_holes
        self.min_height = min_height
        self.min_width = min_width
        self.fill_value = fill_value
        self.p = p

        self.t = A.CoarseDropout(max_holes=self.max_holes, max_height=self.max_height, max_width=self.max_width,
                                 min_holes=self.min_holes, min_height=self.min_height, min_width=self.min_width,
                                 fill_value=self.fill_value, p=self.p)

    def __call__(self, image):
        return self.t(image=image)['image']

    def __repr__(self):
        return self.__class__.__name__ + '(max_holes={0}, max_height={1}, max_width={2}, ' \
                                         'min_holes={3}, min_height={4}, min_width={5}, ' \
                                         'fill_value={6}, p={7})' \
            .format(self.max_holes, self.max_height, self.max_width,
                    self.min_holes, self.min_height, self.min_width, self.fill_value, self.p)

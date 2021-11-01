# -*- coding: utf-8 -*-

"""
@date: 2021/7/23 下午9:27
@file: square_pad.py
@author: zj
@description: 
"""

import cv2
import albumentations as A


class SquarePad(object):
    """
    By filling the shorter edges, the size of the image becomes square

    Args:
        padding_position (Union[str, PositionType]): Position of the image. should be PositionType.CENTER or
            PositionType.TOP_LEFT or PositionType.TOP_RIGHT or PositionType.BOTTOM_LEFT or PositionType.BOTTOM_RIGHT.
            Default: PositionType.CENTER.
        padding_mode (OpenCV flag): OpenCV border mode.
        fill (int, float, list of int, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
        p (float): probability of applying the transform. Default: 1.0.
    """

    def __init__(self, padding_position=A.PadIfNeeded.PositionType.CENTER, padding_mode=cv2.BORDER_CONSTANT,
                 fill=0, p=1.0):
        self.fill = fill
        self.padding_position = padding_position
        self.padding_mode = padding_mode
        self.p = p

        self.t = A.PadIfNeeded(min_width=224, min_height=224, position=self.padding_position,
                               border_mode=self.padding_mode, value=self.fill, p=self.p)

    def __call__(self, image):
        h, w = image.shape[:2]
        if h == w:
            return image

        min_size = h if h > w else w
        self.t.min_width = min_size
        self.t.min_height = min_size
        return self.t(image=image)['image']

    def __repr__(self):
        return self.__class__.__name__ + '(padding_position={0}, padding_mode={1}, fill={2}, p={3})'. \
            format(self.padding_position, self.padding_mode, self.fill, self.p)

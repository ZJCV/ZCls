# -*- coding: utf-8 -*-

"""
@date: 2021/7/23 下午9:27
@file: square_pad.py
@author: zj
@description: 
"""

import numpy as np
import torchvision.transforms.functional as F


class SquarePad(object):
    """
    By filling the shorter edges, the size of the image becomes square
    refer to [How to resize and pad in a torchvision.transforms.Compose()?](https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850)
    """

    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, 'constant')

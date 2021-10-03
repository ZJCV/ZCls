# -*- coding: utf-8 -*-

"""
@date: 2021/10/3 下午1:21
@file: to_tensor.py
@author: zj
@description: 
"""

from albumentations.pytorch.transforms import ToTensorV2


class ToTensor(object):
    """
    Convert image to `torch.Tensor`. The numpy `HWC` image is converted to pytorch `CHW` tensor.
    If the image is in `HW` format (grayscale image), it will be converted to pytorch `HW` tensor.
    """

    def __init__(self, p=1.0):
        self.p = p

        self.t = ToTensorV2(p=self.p)

    def __call__(self, image):
        return self.t(image=image)['image']

    def __repr__(self):
        return self.__class__.__name__ + '(p={0})'.format(self.p)

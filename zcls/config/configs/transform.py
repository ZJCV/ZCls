# -*- coding: utf-8 -*-

"""
@date: 2020/11/25 下午6:51
@file: transform.py
@author: zj
@description: 
"""

from yacs.config import CfgNode as CN
from ztransforms.cls.autoaugment import AutoAugmentPolicy


def add_config(_C):
    # ---------------------------------------------------------------------------- #
    # Transform
    # ---------------------------------------------------------------------------- #
    _C.TRANSFORM = CN()
    _C.TRANSFORM.TRAIN_METHODS = ('Resize', 'CenterCrop', 'ToTensor', 'Normalize')
    _C.TRANSFORM.TEST_METHODS = ('Resize', 'CenterCrop', 'ToTensor', 'Normalize')

    # If size is a sequence like (h, w), output size will be matched to this.
    # If size is an int, smaller edge of the image will be matched to this number.
    # i.e, if height > width, then image will be rescaled to (size * height / width, size).
    _C.TRANSFORM.TRAIN_RESIZE = (224, )
    _C.TRANSFORM.TEST_RESIZE = (224, )

    # Desired output size of the crop.
    # If size is an int instead of sequence like (h, w), a square crop (size, size) is made.
    # If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
    _C.TRANSFORM.TRAIN_CROP = (224, 224)
    _C.TRANSFORM.TEST_CROP = (224, 224)

    # brightness (float or tuple of float (min, max)): How much to jitter brightness.
    #     brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
    #     or the given [min, max]. Should be non negative numbers.
    # contrast (float or tuple of float (min, max)): How much to jitter contrast.
    #     contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
    #     or the given [min, max]. Should be non negative numbers.
    # saturation (float or tuple of float (min, max)): How much to jitter saturation.
    #     saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
    #     or the given [min, max]. Should be non negative numbers.
    # hue (float or tuple of float (min, max)): How much to jitter hue.
    #     hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
    #     Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    # (brightness, contrast, saturation, hue)
    _C.TRANSFORM.ColorJitter = (0.1, 0.1, 0.1, 0.1)

    # AutoAugment policies learned on different datasets.
    _C.TRANSFORM.AUGMENT_POLICY = "imagenet"

    # for AutoAugment, only torch.uint8 image tensors are supported
    # for Normalize, may be happens
    # ValueError: std evaluated to zero after conversion to torch.uint8, leading to division by zero.
    _C.TRANSFORM.IMAGE_DTYPE = 'float32'

    # Sequence of means for each channel.
    _C.TRANSFORM.MEAN = (0.45, 0.45, 0.45)

    # Sequence of standard deviations for each channel.
    _C.TRANSFORM.STD = (0.225, 0.225, 0.225)

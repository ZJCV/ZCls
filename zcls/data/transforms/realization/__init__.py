# -*- coding: utf-8 -*-

"""
@date: 2021/10/3 下午1:25
@file: __init__.py.py
@author: zj
@description: 
"""

__all__ = ['AutoAugment', 'CoarseDropout', 'ColorJitter', 'HorizontalFlip', 'VerticalFlip', 'RandomCrop', 'CenterCrop',
           'Resize', 'Resize2', 'Rotate', 'SquarePad', 'Normalize', 'ToTensor']

from .autoaugment import AutoAugment

from .coarse_dropout import CoarseDropout
from .color_jitter import ColorJitter

from .horizontal_flip import HorizontalFlip
from .vertical_flip import VerticalFlip

from .random_crop import RandomCrop
from .center_crop import CenterCrop

from .resize import Resize
from .rotate import Rotate
from .square_pad import SquarePad

from .normalize import Normalize
from .to_tensor import ToTensor

Resize2 = Resize

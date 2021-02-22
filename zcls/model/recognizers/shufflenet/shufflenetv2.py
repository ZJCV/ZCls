# -*- coding: utf-8 -*-

"""
@date: 2020/12/24 下午7:38
@file: shufflenetv1.py
@author: zj
@description: 
"""

from zcls.model import registry
from ..base_recognizer import BaseRecognizer

"""
G1) Equal channel width minimizes memory access cost (MAC)
G2) Excessive group convolution increases MAC.
G3) Network fragmentation reduces degree of parallelism
G4) Element-wise operations are non-negligible.
"""


@registry.RECOGNIZER.register('ShuffleNetV2')
class ShuffleNetV2(BaseRecognizer):

    def __init__(self, cfg):
        super().__init__(cfg)

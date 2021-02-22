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
Note 1: Empirically g = 3 usually has a proper trade-off between accuracy and actual inference time
Note 2: Comparing ShuffleNet 2× with MobileNet whose complexity are comparable (524 vs. 569 MFLOPs)
"""


@registry.RECOGNIZER.register('ShuffleNetV1')
class ShuffleNetV1(BaseRecognizer):

    def __init__(self, cfg):
        super().__init__(cfg)

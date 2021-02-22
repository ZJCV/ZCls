# -*- coding: utf-8 -*-

"""
@date: 2020/12/2 下午9:38
@file: mobilenetv1.py
@author: zj
@description: 
"""

from zcls.model import registry
from ..base_recognizer import BaseRecognizer


@registry.RECOGNIZER.register('MobileNetV2')
class MobileNetV2(BaseRecognizer):

    def __init__(self, cfg):
        super().__init__(cfg)

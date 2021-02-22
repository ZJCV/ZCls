# -*- coding: utf-8 -*-

"""
@date: 2020/12/24 下午7:38
@file: shufflenetv1.py
@author: zj
@description: 
"""

from zcls.model import registry
from ..base_recognizer import BaseRecognizer


@registry.RECOGNIZER.register('MNASNet')
class MNASNet(BaseRecognizer):

    def __init__(self, cfg):
        super().__init__(cfg)

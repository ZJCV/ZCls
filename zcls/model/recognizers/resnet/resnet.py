# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午2:37
@file: resnet.py
@author: zj
@description: 
"""

from zcls.model import registry
from ..base_recognizer import BaseRecognizer


@registry.RECOGNIZER.register('ResNet')
class ResNet(BaseRecognizer):

    def __init__(self, cfg):
        super().__init__(cfg)

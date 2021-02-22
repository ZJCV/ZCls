# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午2:37
@file: resnet.py
@author: zj
@description: 
"""

from zcls.config.key_word import KEY_OUTPUT
from zcls.model import registry
from ..base_recognizer import BaseRecognizer


@registry.RECOGNIZER.register('ResNet3D')
class ResNet3D(BaseRecognizer):

    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x):
        x = x.unsqueeze(2)

        x = self.backbone(x)
        x = self.head(x)

        return {KEY_OUTPUT: x}

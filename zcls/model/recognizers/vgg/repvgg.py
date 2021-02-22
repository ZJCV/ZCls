# -*- coding: utf-8 -*-

"""
@date: 2021/2/2 下午5:19
@file: repvgg.py
@author: zj
@description: RegVGG，参考[RepVGG/repvgg.py](https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py)
"""

from zcls.model import registry
from ..base_recognizer import BaseRecognizer

@registry.RECOGNIZER.register('RepVGG')
class RepVGG(BaseRecognizer):

    def __init__(self, cfg):
        super().__init__(cfg)

# -*- coding: utf-8 -*-

"""
@date: 2020/11/10 下午5:31
@file: cifar.py
@author: zj
@description: 
"""

from .base_evaluator import BaseEvaluator

class CIFAREvaluator(BaseEvaluator):

    def __init__(self, classes, topk=(1,)):
        super().__init__(classes, topk)

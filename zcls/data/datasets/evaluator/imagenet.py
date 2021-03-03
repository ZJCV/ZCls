# -*- coding: utf-8 -*-

"""
@date: 2021/2/23 下午8:23
@file: fashionmnist.py
@author: zj
@description: 
"""

from .base_evaluator import BaseEvaluator


class ImageNetEvaluator(BaseEvaluator):

    def __init__(self, classes, topk=(1,)):
        super().__init__(classes, topk)

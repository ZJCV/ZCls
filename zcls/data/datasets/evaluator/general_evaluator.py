# -*- coding: utf-8 -*-

"""
@date: 2021/3/10 下午2:07
@file: general_evaluator.py
@author: zj
@description: 
"""

from .base_evaluator import BaseEvaluator


class GeneralEvaluator(BaseEvaluator):

    def __init__(self, classes, top_k=(1,)):
        super().__init__(classes, top_k)

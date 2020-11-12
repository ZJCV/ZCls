# -*- coding: utf-8 -*-

"""
@date: 2020/10/19 上午10:03
@file: base_evaluator.py
@author: zj
@description: 
"""

from abc import ABCMeta, abstractmethod
import torch


class BaseEvaluator(metaclass=ABCMeta):

    def __init__(self, classes):
        self.classes = classes
        self.device = torch.device('cpu')

    @abstractmethod
    def evaluate_train(self, **kwargs):
        pass

    @abstractmethod
    def evaluate_test(self, **kwargs):
        pass

    @abstractmethod
    def get(self):
        pass

    @abstractmethod
    def clean(self):
        pass

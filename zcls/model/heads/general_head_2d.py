# -*- coding: utf-8 -*-

"""
@date: 2020/12/30 下午6:48
@file: general_head_2d.py
@author: zj
@description: 
"""
from abc import ABC

import torch
import torch.nn as nn

from .. import registry


class GeneralHead2D(nn.Module, ABC):

    def __init__(self,
                 feature_dims=1024,
                 dropout_rate=0.,
                 num_classes=1000,
                 bias=True
                 ):
        """
        AvgPool + Dropout + FC
        :param feature_dims: 输入特征维度
        :param dropout_rate: 随机失活概率
        :param num_classes: 类别数
        """
        super(GeneralHead2D, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate, inplace=True)
        self.fc = nn.Linear(feature_dims, num_classes, bias=bias)

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.fc.weight, 0, 0.01)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


@registry.HEAD.register('GeneralHead2D')
def build_general_head_2d(cfg):
    feature_dims = cfg.MODEL.HEAD.FEATURE_DIMS
    num_classes = cfg.MODEL.RECOGNIZER.PRETRAINED_NUM_CLASSES
    dropout_rate = cfg.MODEL.HEAD.DROPOUT_RATE
    bias = cfg.MODEL.HEAD.BIAS

    return GeneralHead2D(feature_dims=feature_dims,
                         num_classes=num_classes,
                         dropout_rate=dropout_rate,
                         bias=bias)

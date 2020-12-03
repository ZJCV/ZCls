# -*- coding: utf-8 -*-

"""
@date: 2020/12/3 下午8:17
@file: mobilenetv1_head.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn


class MobileNetV1Head(nn.Module):

    def __init__(self,
                 # 输入特征维度
                 feature_dims=1024,
                 # 类别数
                 num_classes=1000,
                 ):
        super(MobileNetV1Head, self).__init__()

        self.avgpool = nn.AvgPool2d((7, 7))
        self.fc = nn.Linear(feature_dims, num_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

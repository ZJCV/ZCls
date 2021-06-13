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

from zcls.model.conv_helper import get_conv
from zcls.model.act_helper import get_act
from .. import registry


class MobileNetV3Head(nn.Module, ABC):

    def __init__(self,
                 feature_dims=960,
                 inner_dims=1280,
                 dropout_rate=0.,
                 num_classes=1000,
                 conv_layer=None,
                 act_layer=None
                 ):
        """
        :param feature_dims: 输入特征维度
        :param inner_dims: 中间特征维度
        :param dropout_rate: 随机失活概率
        :param num_classes: 类别数
        :param conv_layer: 卷积层类型
        :param act_layer: 激活层类型
        """
        super(MobileNetV3Head, self).__init__()

        if conv_layer is None:
            conv_layer = nn.Conv2d
        if act_layer is None:
            act_layer = nn.Hardswish

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = conv_layer(feature_dims, inner_dims, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.conv2 = conv_layer(inner_dims, num_classes, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.act = act_layer(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate, inplace=True)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv1(x)
        x = self.act(x)

        x = self.dropout(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        return x


@registry.HEAD.register('MobileNetV3')
def build_mbv3_head(cfg):
    feature_dims = cfg.MODEL.HEAD.FEATURE_DIMS
    inner_dims = cfg.MODEL.HEAD.INNER_DIMS
    dropout_rate = cfg.MODEL.HEAD.DROPOUT_RATE
    num_classes = cfg.MODEL.RECOGNIZER.PRETRAINED_NUM_CLASSES
    conv_layer = get_conv(cfg)
    act_layer = get_act(cfg)

    return MobileNetV3Head(feature_dims=feature_dims,
                           inner_dims=inner_dims,
                           dropout_rate=dropout_rate,
                           num_classes=num_classes,
                           conv_layer=conv_layer,
                           act_layer=act_layer)

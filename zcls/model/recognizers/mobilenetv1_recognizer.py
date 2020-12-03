# -*- coding: utf-8 -*-

"""
@date: 2020/12/2 下午9:38
@file: mobilenetv1_recognizer.py
@author: zj
@description: 
"""

import torch.nn as nn

from ..backbones.mobilenetv1_backbone import MobileNetV1Backbone
from ..heads.mobilenetv1_head import MobileNetV1Head
from ..norm_helper import get_norm, freezing_bn


class MobileNetV1Recognizer(nn.Module):

    def __init__(self,
                 # 输入通道数
                 inplanes=3,
                 # 第一个卷积层通道数
                 base_channel=32,
                 # 后续各深度卷积通道数
                 channels=(64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024),
                 # 卷积步长
                 strides=(1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 2),
                 # Head输入特征数
                 feature_dims=1024,
                 # 类别数
                 num_classes=1000,
                 # 固定BN
                 fix_bn=False,
                 # 仅训练第一层BN
                 partial_bn=False,
                 # 卷积层类型
                 conv_layer=None,
                 # 归一化层类型
                 norm_layer=None,
                 # 激活层类型
                 act_layer=None
                 ):
        super(MobileNetV1Recognizer, self).__init__()
        self.num_classes = num_classes
        self.fix_bn = fix_bn
        self.partial_bn = partial_bn

        self.backbone = MobileNetV1Backbone(
            inplanes=inplanes,
            base_channel=base_channel,
            channels=channels,
            strides=strides,
            conv_layer=conv_layer,
            norm_layer=norm_layer,
            act_layer=act_layer
        )
        self.head = MobileNetV1Head(
            feature_dims=feature_dims,
            num_classes=1000
        )

    def train(self, mode: bool = True):
        super(MobileNetV1Recognizer, self).train(mode=mode)

        if mode and (self.partial_bn or self.fix_bn):
            freezing_bn(self, partial_bn=self.partial_bn)

        return self

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)

        return {'probs': x}

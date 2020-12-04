# -*- coding: utf-8 -*-

"""
@date: 2020/12/4 上午11:39
@file: mobilenetv2_inverted_residual.py
@author: zj
@description: MobileNetV2 反向残差块
"""

import torch.nn as nn


class MobileNetV2InvertedResidual(nn.Module):
    """
    MobileNetV2的反向残差块由一个膨胀卷积和一个深度可分离卷积组成
    参考torchvision实现:
    1. 当膨胀率大于1时，执行膨胀卷积操作；
    2. 当深度卷积步长为1且输入/输出通道数相同时，执行残差连接
    3. 反向残差块的最后不执行激活操作
    """

    def __init__(self,
                 # 输入通道数
                 inplanes,
                 # 输出通道数
                 planes,
                 # 膨胀因子
                 t=1,
                 # 卷积层步长
                 stride=1,
                 # 卷积层零填充
                 padding=1,
                 # 卷积层类型
                 conv_layer=None,
                 # 归一化层类型
                 norm_layer=None,
                 # 激活层类型
                 act_layer=None,
                 ):
        super(MobileNetV2InvertedResidual, self).__init__()

        if conv_layer is None:
            conv_layer = nn.Conv2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU6

        self.conv_layer = conv_layer
        self.norm_layer = norm_layer
        self.act_layer = act_layer

        # 计算隐藏层输入通道数
        hidden_planes = int(t * inplanes)
        features = list()
        if t != 1:
            features.append(nn.Sequential(
                self.conv_layer(inplanes, hidden_planes, kernel_size=1, stride=1, bias=False),
                self.norm_layer(hidden_planes),
                self.act_layer(inplace=True)
            ))

        # 深度卷积
        features.append(nn.Sequential(
            self.conv_layer(hidden_planes, hidden_planes, kernel_size=3, stride=stride, padding=padding,
                            groups=hidden_planes, bias=False),
            self.norm_layer(hidden_planes),
            self.act_layer(inplace=True)
        ))

        # 逐点卷积
        features.append(nn.Sequential(
            self.conv_layer(hidden_planes, planes, kernel_size=1, stride=1, bias=False),
            self.norm_layer(planes)
        ))

        self.conv = nn.Sequential(*features)
        self.use_res_connect = stride == 1 and inplanes == planes

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

# -*- coding: utf-8 -*-

"""
@date: 2021/6/3 下午8:26
@file: ghost_bottleneck.py
@author: zj
@description: 
"""

import torch.nn as nn

from zcls.model.attention_helper import make_attention_block
from .ghost_module import GhostModule


class GhostBottleneck(nn.Module):

    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 stride=1,
                 kernel_size=3,
                 with_attention=False,
                 reduction=4,
                 attention_type='SqueezeAndExcitationBlock2D'
                 ):
        """
        when stride=1, bottleneck=Add(
                                    Input,
                                    BN(GhostModule(
                                        ReLU(BN(
                                            GhostModule(Input)
                                        ))
                                    ))
                                    )
        when stride=2, bottleneck=Add(
                                    Input,
                                    BN(GhostModule(
                                        BN(DWConv(
                                            ReLU(BN(
                                                GhostModule(Input)
                                            ))
                                        ))
                                    ))
                                    )
        if assert squeeze-and-excitation module, use it before ghost2
        :param in_channels: 输入通道数
        :param mid_channels: 膨胀通道数
        :param out_channels: 输出通道数
        :param stride: 步长，是否执行下采样操作
        :param kernel_size: 深度卷积核大小
        :param with_attention: 是否执行注意力模块
        :param reduction: 衰减率
        :param attention_type: 注意力类型
        """
        super(GhostBottleneck, self).__init__()

        # Point-wise expansion
        self.ghost1 = GhostModule(in_channels, mid_channels, is_act=True)

        self.stride = stride
        # Depth-wise convolution
        if self.stride > 1:
            padding = (kernel_size - 1) // 2
            self.conv_dw = nn.Conv2d(mid_channels, mid_channels, (kernel_size, kernel_size), stride=(stride, stride),
                                     padding=(padding, padding), groups=mid_channels, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_channels)

        # Squeeze-and-excitation
        if with_attention:
            self.se = make_attention_block(mid_channels,
                                           reduction,
                                           attention_type,
                                           sigmoid_type='HSigmoid',
                                           is_round=True,
                                           round_nearest=4,
                                           bias=True,
                                           )
        self.with_attention = with_attention

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_channels, out_channels, is_act=False)

        # shortcut
        if in_channels == out_channels and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            padding = (kernel_size - 1) // 2
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, (kernel_size, kernel_size), stride=(stride, stride),
                          padding=(padding, padding), groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, (1, 1), stride=(1, 1), padding=(0, 0), bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.with_attention:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x += self.shortcut(residual)
        return x

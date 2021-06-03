# -*- coding: utf-8 -*-

"""
@date: 2020/12/24 下午8:04
@file: shufflenetv1_unit.py
@author: zj
@description: 
"""

from abc import ABC

import torch
import torch.nn as nn


class ShuffleNetV1Unit(nn.Module, ABC):

    def __init__(self,
                 in_channels,
                 out_channels,
                 groups,
                 stride,
                 downsample=None,
                 with_group=True,
                 conv_layer=None,
                 norm_layer=None,
                 act_layer=None,
                 ):
        """
        In paper https://arxiv.org/abs/1707.01083
        when stride=1, Unit = ReLU(Add(
                                Input,
                                BN(1x1 GConv(
                                    BN(3x3 DWConv(
                                        Channel Shuffle(
                                            ReLU(BN(1x1 GConv(Input)))
                                        )
                                    ))
                                ))
                              ));
        when stride=2, Unit = ReLU(Concat(
                                S=2 K=3x3 AvgPool(Input),
                                BN(1x1 GConv(
                                  BN(S=2 K=3x3 DWConv(
                                    Channel Shuffle(
                                        ReLU(BN(1x1 GConv(Input)))
                                    )
                                  ))
                                ))
                              ));
        In official realization [ShuffleNet-Series/ShuffleNetV1/blocks.py](https://github.com/megvii-model/ShuffleNet-Series/blob/master/ShuffleNetV1/blocks.py)
        There are two differences with the paper description
        1. use channel shuffle after 3x3 DWConv;
        2. when stride=2, not use activation function for identity mapping
                                Unit = Concat(
                                    S=2 K=3x3 AvgPool(Input),
                                    ReLU(BN(1x1 GConv(
                                        Channel Shuffle(
                                            BN(S=2 K=3x3 DWConv(
                                                ReLU(BN(1x1 GConv(Input)))
                                            ))
                                        )
                                    )))
                                );
        for first:
        * [Why is ShuffleNetV1 different from description in paper ? #16](https://github.com/megvii-model/ShuffleNet-Series/issues/16)
        * [shufflenetv1的一个问题 #40](https://github.com/megvii-model/ShuffleNet-Series/issues/40)
        for second:
        * [A mismatch in shufflenetv1 about ReLU #53](https://github.com/megvii-model/ShuffleNet-Series/issues/53)
        :param in_channels: 输入通道
        :param out_channels: 输出通道
        :param groups: 分组数
        :param stride: 步长
        :param downsample: 作用于shortcut path
        :param with_group: 是否对第一个1x1逐点卷积应用分组
        :param conv_layer: 卷积层类型
        :param norm_layer: 归一化层类型
        :param act_layer: 激活层类型
        """
        super().__init__()
        assert out_channels % groups == 0

        if conv_layer is None:
            conv_layer = nn.Conv2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU

        mid_channels = out_channels // 4
        out_channels = out_channels if stride == 1 else out_channels - in_channels
        self.conv1 = conv_layer(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False,
                                groups=groups if with_group else 1)
        self.norm1 = norm_layer(mid_channels)

        self.conv2 = conv_layer(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False,
                                groups=mid_channels)
        self.norm2 = norm_layer(mid_channels)

        self.conv3 = conv_layer(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False,
                                groups=groups)
        self.norm3 = norm_layer(out_channels)

        self.act = act_layer(inplace=True)
        self.down_sample = downsample
        self.groups = groups
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.groups > 1:
            out = self.channel_shuffle(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.stride == 2:
            out = torch.cat((self.down_sample(identity), out), dim=1)
        else:
            out = self.act(out + identity)
        return out

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % self.groups == 0
        group_channels = num_channels // self.groups

        x = x.reshape(batchsize, group_channels, self.groups, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batchsize, num_channels, height, width)

        return x

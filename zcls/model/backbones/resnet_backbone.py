# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午7:04
@file: resnet_backbone.py
@author: zj
@description: 
"""

import torch.nn as nn

from ..backbones.basicblock import BasicBlock
from ..backbones.bottleneck import Bottleneck


class ResNetBackbone(nn.Module):

    def __init__(self,
                 # 输入通道数
                 inplanes=3,
                 # 基础通道数,
                 base_planes=64,
                 # 每一层通道数
                 layer_planes=(64, 128, 256, 512),
                 # 每一层块个数
                 layer_blocks=(2, 2, 2, 2),
                 # 是否执行空间下采样
                 downsamples=(0, 1, 1, 1),
                 # 块类型
                 block_layer=None,
                 # 卷积层类型
                 conv_layer=None,
                 # 归一化层类型
                 norm_layer=None,
                 # 激活层类型
                 act_layer=None,
                 # 零初始化残差连接
                 zero_init_residual=False
                 ):
        super(ResNetBackbone, self).__init__()

        if block_layer is None:
            block_layer = BasicBlock
        if conv_layer is None:
            conv_layer = nn.Conv2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU

        self.inplanes = inplanes
        self.base_planes = base_planes
        self.layer_planes = layer_planes
        self.layer_blocks = layer_blocks
        self.downsamples = downsamples
        self.block_layer = block_layer
        self.conv_layer = conv_layer
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        self.zero_init_residual = zero_init_residual

        self._make_stem()
        self.inplanes = self.base_planes
        for i in range(len(self.layer_blocks)):
            res_layer = self._make_res_layer(self.inplanes,
                                             self.layer_planes[i],
                                             self.layer_blocks[i],
                                             self.downsamples[i],
                                             self.block_layer,
                                             self.conv_layer,
                                             self.norm_layer,
                                             self.act_layer
                                             )
            self.inplanes = self.layer_planes[i] * self.block_layer.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)

        self._init_weights(self.zero_init_residual)

    def _make_stem(self):
        self.conv1 = self.conv_layer(self.inplanes, self.base_planes, kernel_size=(7, 7), stride=2, padding=3,
                                     bias=False)
        self.bn1 = self.norm_layer(self.base_planes)
        self.relu = self.act_layer(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _make_res_layer(self,
                        # 输入通道数
                        inplanes,
                        # 输出通道数
                        planes,
                        # 块个数
                        block_num,
                        # 是否执行空间下采样
                        is_downsample,
                        # 块类型
                        block_layer,
                        # 卷积层类型
                        conv_layer,
                        # 归一化层类型
                        norm_layer,
                        # 激活层类型
                        act_layer,
                        ):
        stride = 2 if is_downsample else 1
        expansion = self.block_layer.expansion
        if is_downsample or inplanes != planes * expansion:
            downsample = nn.Sequential(
                conv_layer(inplanes, planes * expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * expansion),
            )
        else:
            downsample = None

        blocks = list()
        blocks.append(block_layer(
            inplanes, planes, stride, downsample, conv_layer, norm_layer, act_layer))
        inplanes = planes * expansion

        for i in range(1, block_num):
            blocks.append(block_layer(inplanes, planes, 1, None, conv_layer, norm_layer, act_layer))
        return nn.Sequential(*blocks)

    def _init_weights(self, zero_init_residual):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

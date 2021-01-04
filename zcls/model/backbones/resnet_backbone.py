# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午7:04
@file: resnet_backbone.py
@author: zj
@description: 
"""
from abc import ABC

import torch.nn as nn

from ..backbones.basicblock import BasicBlock
from ..backbones.bottleneck import Bottleneck


class ResNetBackbone(nn.Module, ABC):
    """
    参考Torchvision实现，适用于torchvision预训练模型加载
    """

    def __init__(self,
                 # 输入通道数
                 in_planes=3,
                 # 基础通道数,
                 base_planes=64,
                 # 每一层通道数
                 layer_planes=(64, 128, 256, 512),
                 # 每一层块个数
                 layer_blocks=(2, 2, 2, 2),
                 # 是否执行空间下采样
                 down_samples=(0, 1, 1, 1),
                 # cardinality
                 groups=1,
                 # 每组的宽度
                 width_per_group=64,
                 # 是否使用注意力模块
                 with_attentions=(0, 0, 0, 0),
                 # 衰减率
                 reduction=16,
                 # 注意力模块类型
                 attention_type='SqueezeAndExcitationBlock2D',
                 # 块类型
                 block_layer=None,
                 # 卷积层类型
                 conv_layer=None,
                 # 归一化层类型
                 norm_layer=None,
                 # 激活层类型
                 act_layer=None,
                 # 零初始化残差连接
                 zero_init_residual=False,
                 # 其他参数
                 **kwargs
                 ):
        super(ResNetBackbone, self).__init__()
        assert len(layer_planes) == len(layer_blocks) == len(down_samples) == len(with_attentions)

        if block_layer is None:
            block_layer = BasicBlock
        if conv_layer is None:
            conv_layer = nn.Conv2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU

        self._make_stem(in_planes, base_planes, conv_layer, norm_layer, act_layer)
        in_planes = base_planes
        for i in range(len(layer_blocks)):
            res_layer = self._make_res_layer(in_planes,
                                             layer_planes[i],
                                             layer_blocks[i],
                                             down_samples[i],
                                             groups,
                                             width_per_group,
                                             with_attentions[i],
                                             reduction,
                                             attention_type,
                                             block_layer,
                                             conv_layer,
                                             norm_layer,
                                             act_layer
                                             )
            in_planes = layer_planes[i] * block_layer.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)

        self.init_weights(zero_init_residual)

    def _make_stem(self,
                   # 输入通道数
                   in_planes,
                   # Stem块输出通道数
                   base_planes,
                   # 卷积层
                   conv_layer,
                   # 池化层
                   norm_layer,
                   # 激活层
                   act_layer
                   ):
        self.conv1 = conv_layer(in_planes, base_planes, kernel_size=(7, 7), stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(base_planes)
        self.relu = act_layer(inplace=True)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _make_res_layer(self,
                        # 输入通道数
                        in_planes,
                        # 输出通道数
                        out_planes,
                        # 块个数
                        block_num,
                        # 是否执行空间下采样
                        with_sample,
                        # cardinality
                        groups,
                        # 每组的宽度
                        width_per_group,
                        # 是否使用注意力模块
                        with_attention,
                        # 衰减率
                        reduction,
                        # 注意力模块类型
                        attention_type,
                        # 块类型
                        block_layer,
                        # 卷积层类型
                        conv_layer,
                        # 归一化层类型
                        norm_layer,
                        # 激活层类型
                        act_layer,
                        # 其他参数
                        **kwargs
                        ):
        assert isinstance(with_attention, (int, tuple))
        assert with_attention in (0, 1) if isinstance(with_attention, int) else len(with_attention) == block_num
        with_attentions = with_attention if isinstance(with_attention, tuple) else [with_attention] * block_num

        stride = 2 if with_sample else 1
        expansion = block_layer.expansion
        if with_sample or in_planes != out_planes * expansion:
            down_sample = nn.Sequential(
                conv_layer(in_planes, out_planes * expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(out_planes * expansion),
            )
        else:
            down_sample = None

        blocks = list()
        blocks.append(block_layer(
            in_planes, out_planes, stride, down_sample,
            groups, width_per_group,
            with_attentions[0], reduction, attention_type,
            conv_layer, norm_layer, act_layer, **kwargs))
        in_planes = out_planes * expansion

        stride = 1
        down_sample = None
        for i in range(1, block_num):
            blocks.append(block_layer(
                in_planes, out_planes, stride, down_sample,
                groups, width_per_group,
                with_attentions[i], reduction, attention_type,
                conv_layer, norm_layer, act_layer, **kwargs))
        return nn.Sequential(*blocks)

    def init_weights(self, zero_init_residual):
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
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

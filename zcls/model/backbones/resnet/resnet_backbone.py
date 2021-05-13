# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午7:04
@file: resnet_backbone.py
@author: zj
@description: 
"""

from abc import ABC
import torch.nn as nn

from zcls.model import registry
from zcls.model.conv_helper import get_conv
from zcls.model.norm_helper import get_norm
from zcls.model.act_helper import get_act
from .basicblock import BasicBlock
from .bottleneck import Bottleneck
from .sknet_block import SKNetBlock

arch_settings = {
    # name: (Block, Layer_planes, groups, width_per_group)
    'resnet18': (BasicBlock, (2, 2, 2, 2), 1, 64),
    'resnet34': (BasicBlock, (3, 4, 6, 3), 1, 64),
    'resnet50': (Bottleneck, (3, 4, 6, 3), 1, 64),
    'resnet101': (Bottleneck, (3, 4, 23, 3), 1, 64),
    'resnet152': (Bottleneck, (3, 8, 36, 3), 1, 64),
    'resnext50_32x4d': (Bottleneck, (3, 4, 6, 3), 32, 4),
    'resnext101_32x8d': (Bottleneck, (3, 4, 23, 3), 32, 8),
    # name: (Block, Layer_planes, groups, width_per_group)
    'sknet50': (SKNetBlock, (3, 4, 6, 3), 32, 4),
}


def make_stem(in_channels,
              base_channels,
              conv_layer,
              norm_layer,
              act_layer
              ):
    """

    :param in_channels: 输入通道数
    :param base_channels: Stem块输出通道数
    :param conv_layer: 卷积层
    :param norm_layer: 池化层
    :param act_layer: 激活层
    :return:
    """
    return nn.Sequential(
        conv_layer(in_channels, base_channels, kernel_size=(7, 7), stride=2, padding=3, bias=False),
        norm_layer(base_channels),
        act_layer(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )


def make_res_layer(in_planes,
                   out_planes,
                   block_num,
                   with_sample,
                   groups,
                   width_per_group,
                   with_attention,
                   reduction,
                   attention_type,
                   block_layer,
                   conv_layer,
                   norm_layer,
                   act_layer,
                   **kwargs
                   ):
    """

    :param in_planes: 输入通道数
    :param out_planes: 输出通道数
    :param block_num: 块个数
    :param with_sample: 是否执行空间下采样
    :param groups: cardinality
    :param width_per_group: 每组的宽度
    :param with_attention: 是否使用注意力模块
    :param reduction: 衰减率
    :param attention_type: 注意力模块类型
    :param block_layer: 块类型
    :param conv_layer: 卷积层类型
    :param norm_layer: 归一化层类型
    :param act_layer: 激活层类型
    :param kwargs: 其他参数
    :return:
    """
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


class ResNetBackbone(nn.Module, ABC):

    def __init__(self,
                 in_channels=3,
                 base_channels=64,
                 layer_channels=(64, 128, 256, 512),
                 layer_blocks=(2, 2, 2, 2),
                 down_samples=(0, 1, 1, 1),
                 groups=1,
                 width_per_group=64,
                 with_attentions=(0, 0, 0, 0),
                 reduction=16,
                 attention_type='SqueezeAndExcitationBlock2D',
                 block_layer=None,
                 conv_layer=None,
                 norm_layer=None,
                 act_layer=None,
                 zero_init_residual=False,
                 **kwargs
                 ):
        """
        参考Torchvision实现，适用于torchvision预训练模型加载
        :param in_channels: 输入通道数
        :param base_channels: 基础通道数
        :param layer_channels: 每一层通道数
        :param layer_blocks: 每一层块个数
        :param down_samples: 是否执行空间下采样
        :param groups: cardinality
        :param width_per_group: 每组的宽度
        :param with_attentions: 是否使用注意力模块
        :param reduction: 衰减率
        :param attention_type: 注意力模块类型
        :param block_layer: 块类型
        :param conv_layer: 卷积层类型
        :param norm_layer: 归一化层类型
        :param act_layer: 激活层类型
        :param zero_init_residual: 零初始化残差连接
        :param kwargs: 其他参数
        """
        super(ResNetBackbone, self).__init__()
        assert len(layer_channels) == len(layer_blocks) == len(down_samples) == len(with_attentions)

        if block_layer is None:
            block_layer = BasicBlock
        if conv_layer is None:
            conv_layer = nn.Conv2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU

        self.stem = make_stem(in_channels, base_channels, conv_layer, norm_layer, act_layer)
        in_channels = base_channels
        for i in range(len(layer_blocks)):
            res_layer = make_res_layer(in_channels,
                                       layer_channels[i],
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
                                       act_layer,
                                       **kwargs
                                       )
            in_channels = layer_channels[i] * block_layer.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
        self.layer_num = len(layer_blocks)

        self.init_weights(zero_init_residual)

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
        x = self.stem(x)

        for i in range(self.layer_num):
            layer = self.__getattr__(f'layer{i + 1}')
            x = layer(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


@registry.Backbone.register('ResNet')
def build_resnet_backbone(cfg):
    arch = cfg.MODEL.BACKBONE.ARCH
    in_planes = cfg.MODEL.BACKBONE.IN_PLANES
    base_planes = cfg.MODEL.BACKBONE.BASE_PLANES
    layer_planes = cfg.MODEL.BACKBONE.LAYER_PLANES
    down_samples = cfg.MODEL.BACKBONE.DOWNSAMPLES
    conv_layer = get_conv(cfg)
    norm_layer = get_norm(cfg)
    act_layer = get_act(cfg)
    zero_init_residual = cfg.MODEL.RECOGNIZER.ZERO_INIT_RESIDUAL
    # for attention
    with_attentions = cfg.MODEL.ATTENTION.WITH_ATTENTIONS
    reduction = cfg.MODEL.ATTENTION.REDUCTION
    attention_type = cfg.MODEL.ATTENTION.ATTENTION_TYPE

    block_layer, layer_blocks, groups, width_per_group = arch_settings[arch]

    return ResNetBackbone(
        in_channels=in_planes,
        base_channels=base_planes,
        layer_channels=layer_planes,
        layer_blocks=layer_blocks,
        down_samples=down_samples,
        groups=groups,
        width_per_group=width_per_group,
        with_attentions=with_attentions,
        reduction=reduction,
        attention_type=attention_type,
        block_layer=block_layer,
        conv_layer=conv_layer,
        norm_layer=norm_layer,
        act_layer=act_layer,
        zero_init_residual=zero_init_residual,
    )

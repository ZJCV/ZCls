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
from .resnest_block import ResNeStBlock

arch_settings = {
    # name: (Block, Layer_planes, groups, width_per_group)
    'resnet18': (BasicBlock, (2, 2, 2, 2), 1, 64),
    'resnet34': (BasicBlock, (3, 4, 6, 3), 1, 64),
    'resnet50': (Bottleneck, (3, 4, 6, 3), 1, 64),
    'resnet101': (Bottleneck, (3, 4, 23, 3), 1, 64),
    'resnet152': (Bottleneck, (3, 8, 36, 3), 1, 64),
    'resnext50_32x4d': (Bottleneck, (3, 4, 6, 3), 32, 4),
    'resnedxt101_32x8d': (Bottleneck, (3, 4, 23, 3), 32, 8),
    # name: (Block, Layer_planes, groups, width_per_group)
    'sknet50': (SKNetBlock, (3, 4, 6, 3), 32, 4),
    # name: (Block, Layer_planes, radix, groups, width_per_group)
    'resnest50_1s1x64d': (ResNeStBlock, (3, 4, 6, 3), 1, 1, 64),
    'resnest50_fast_2s1x64d': (ResNeStBlock, (3, 4, 6, 3), 2, 1, 64),
    'resnest50_4s1x64d': (ResNeStBlock, (3, 4, 6, 3), 4, 1, 64),
    'resnest50_2s2x40d': (ResNeStBlock, (3, 4, 6, 3), 2, 2, 40),
    'resnest50_fast_2s2x40d': (ResNeStBlock, (3, 4, 6, 3), 2, 2, 40),
    'resnest50': (ResNeStBlock, (3, 4, 6, 3), 2, 1, 64),
    'resnest101': (ResNeStBlock, (3, 4, 23, 3), 2, 1, 64),
    'resnest200': (ResNeStBlock, (3, 24, 36, 3), 2, 1, 64),
    'resnest269': (ResNeStBlock, (3, 30, 48, 8), 2, 1, 64),
}


def make_stem(in_channels,
              base_channels,
              conv_layer,
              norm_layer,
              act_layer
              ):
    """

    :param in_channels: 输入通道数
    :param base_channels: 卷积层输出通道数
    :param conv_layer: 卷积层
    :param norm_layer: 池化层
    :param act_layer: 激活层
    :return:
    """
    inner_planes = base_channels // 2
    return nn.Sequential(
        conv_layer(in_channels, inner_planes, kernel_size=3, stride=2, padding=1, bias=False),
        norm_layer(inner_planes),
        act_layer(inplace=True),
        conv_layer(inner_planes, inner_planes, kernel_size=3, stride=1, padding=1, bias=False),
        norm_layer(inner_planes),
        act_layer(inplace=True),
        conv_layer(inner_planes, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
        norm_layer(base_channels),
        act_layer(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )


def make_res_layer(in_channels,
                   out_channels,
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
                   use_avg,
                   fast_avg,
                   **kwargs
                   ):
    """

    :param in_channels: 输入通道数
    :param out_channels: 输出通道数
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
    :param use_avg: 是否使用AvgPool进行下采样
    :param fast_avg: 在3x3之前执行下采样操作
    :param kwargs: 其他参数
    :return:
    """
    assert isinstance(with_attention, (int, tuple))
    assert with_attention in (0, 1) if isinstance(with_attention, int) else len(with_attention) == block_num
    with_attentions = with_attention if isinstance(with_attention, tuple) else [with_attention] * block_num

    stride = 2 if with_sample else 1
    expansion = block_layer.expansion
    if with_sample:
        down_sample = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            conv_layer(in_channels, out_channels * expansion, kernel_size=1, stride=1, bias=False),
            norm_layer(out_channels * expansion),
        )
    elif in_channels != out_channels * expansion:
        down_sample = nn.Sequential(
            conv_layer(in_channels, out_channels * expansion, kernel_size=1, stride=1, bias=False),
            norm_layer(out_channels * expansion),
        )
    else:
        down_sample = None

    blocks = list()
    blocks.append(block_layer(
        in_channels, out_channels, stride, down_sample,
        groups, width_per_group,
        with_attentions[0], reduction, attention_type,
        conv_layer, norm_layer, act_layer,
        use_avg=use_avg, fast_avg=fast_avg,
        **kwargs))
    in_channels = out_channels * expansion

    stride = 1
    down_sample = None
    for i in range(1, block_num):
        blocks.append(block_layer(
            in_channels, out_channels, stride, down_sample,
            groups, width_per_group,
            with_attentions[i], reduction, attention_type,
            conv_layer, norm_layer, act_layer,
            use_avg=use_avg, fast_avg=fast_avg,
            **kwargs))
    return nn.Sequential(*blocks)


class ResNetDBackbone(nn.Module, ABC):

    def __init__(self,
                 in_channels=3,
                 base_channels=64,
                 layer_channels=(64, 128, 256, 512),
                 layer_blocks=(2, 2, 2, 2),
                 downsamples=(0, 1, 1, 1),
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
                 use_avg=False,
                 fast_avg=False,
                 **kwargs
                 ):
        """
        参考：[Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/abs/1812.01187)
        :param in_channels: 输入通道数
        :param base_channels: 基础通道数
        :param layer_channels: 每一层通道数
        :param layer_blocks: 每一层块个数
        :param downsamples: 是否执行空间下采样
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
        :param use_avg: 是否使用AvgPool进行下采样
        :param fast_avg: 在3x3之前执行下采样操作
        :param kwargs: 其他参数
        """
        super(ResNetDBackbone, self).__init__()
        assert len(layer_channels) == len(layer_blocks) == len(downsamples) == len(with_attentions)

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
                                       downsamples[i],
                                       groups,
                                       width_per_group,
                                       with_attentions[i],
                                       reduction,
                                       attention_type,
                                       block_layer,
                                       conv_layer,
                                       norm_layer,
                                       act_layer,
                                       use_avg,
                                       fast_avg,
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


@registry.Backbone.register('ResNetD')
def build_resnet_d_backbone(cfg):
    arch = cfg.MODEL.BACKBONE.ARCH
    in_planes = cfg.MODEL.BACKBONE.IN_PLANES
    base_planes = cfg.MODEL.BACKBONE.BASE_PLANES
    layer_planes = cfg.MODEL.BACKBONE.LAYER_PLANES
    down_samples = cfg.MODEL.BACKBONE.DOWNSAMPLES
    conv_layer = get_conv(cfg)
    norm_layer = get_norm(cfg)
    act_layer = get_act(cfg)
    zero_init_residual = cfg.MODEL.RECOGNIZER.ZERO_INIT_RESIDUAL
    use_avg = cfg.MODEL.BACKBONE.USE_AVG
    # fast_avg = cfg.MODEL.BACKBONE.FAST_AVG

    # for attention
    with_attentions = cfg.MODEL.ATTENTION.WITH_ATTENTIONS
    reduction = cfg.MODEL.ATTENTION.REDUCTION
    attention_type = cfg.MODEL.ATTENTION.ATTENTION_TYPE

    radix = 1
    fast_avg = True if 'fast' in arch else False
    if 'resnest' in arch:
        block_layer, layer_blocks, radix, groups, width_per_group = arch_settings[arch]
    else:
        block_layer, layer_blocks, groups, width_per_group = arch_settings[arch]

    return ResNetDBackbone(
        in_channels=in_planes,
        base_channels=base_planes,
        layer_channels=layer_planes,
        layer_blocks=layer_blocks,
        downsamples=down_samples,
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
        radix=radix,
        use_avg=use_avg,
        fast_avg=fast_avg,
    )

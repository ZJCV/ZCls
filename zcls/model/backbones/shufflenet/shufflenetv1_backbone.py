# -*- coding: utf-8 -*-

"""
@date: 2020/12/3 下午8:52
@file: mobilenetv2_backbone.py
@author: zj
@description: 
"""

from abc import ABC
import copy
import torch.nn as nn

from zcls.model import registry
from zcls.model.init_helper import init_weights
from zcls.model.conv_helper import get_conv
from zcls.model.norm_helper import get_norm
from zcls.model.act_helper import get_act
from .shufflenetv1_unit import ShuffleNetV1Unit

arch_settings = {
    # block_layer, groups, base_channel, stage_channels, layer_blocks, width_multiplier
    'shufflenetv1_3g2x': (ShuffleNetV1Unit, 3, 48, [240, 480, 960], (4, 8, 4), 2.0),
    'shufflenetv1_3g1_5x': (ShuffleNetV1Unit, 3, 24, [240, 480, 960], (4, 8, 4), 1.5),
    'shufflenetv1_3g1x': (ShuffleNetV1Unit, 3, 24, [240, 480, 960], (4, 8, 4), 1.0),
    'shufflenetv1_3g0_5x': (ShuffleNetV1Unit, 3, 12, [240, 480, 960], (4, 8, 4), 0.5),
    'shufflenetv1_8g2x': (ShuffleNetV1Unit, 8, 48, [384, 768, 1536], (4, 8, 4), 2.0),
    'shufflenetv1_8g1_5x': (ShuffleNetV1Unit, 8, 24, [384, 768, 1536], (4, 8, 4), 1.5),
    'shufflenetv1_8g1x': (ShuffleNetV1Unit, 8, 24, [384, 768, 1536], (4, 8, 4), 1.0),
    'shufflenetv1_8g0_5x': (ShuffleNetV1Unit, 8, 16, [384, 768, 1536], (4, 8, 4), 0.5),
}


def make_stage(in_channels,
               out_channels,
               groups,
               block_num,
               with_downsample,
               with_groups,
               block_layer,
               conv_layer,
               norm_layer,
               act_layer,
               ):
    """
    :param in_channels: 输入通道数
    :param out_channels: 输出通道数
    :param groups: 分组数
    :param block_num: 块个数
    :param with_downsample: 是否执行空间下采样
    :param with_groups: 是否对第一个1x1逐点卷积执行分组操作
    :param block_layer: 块类型
    :param conv_layer: 卷积层类型
    :param norm_layer: 归一化层类型
    :param act_layer: 激活层类型
    :return:
    """
    with_groups = with_groups if isinstance(with_groups, tuple) else [with_groups] * block_num
    assert len(with_groups) == block_num
    stride = 2 if with_downsample else 1
    down_sample = nn.AvgPool2d(3, stride=stride, padding=1) if stride == 2 else None

    blocks = list()
    blocks.append(block_layer(
        in_channels, out_channels, groups, stride, down_sample, with_groups[0], conv_layer, norm_layer, act_layer))
    in_channels = out_channels

    stride = 1
    down_sample = None
    for i in range(1, block_num):
        blocks.append(block_layer(
            in_channels, out_channels, groups, stride, down_sample, True, conv_layer, norm_layer, act_layer))
    return nn.Sequential(*blocks)


def make_stem(in_planes,
              base_planes,
              conv_layer,
              norm_layer,
              act_layer
              ):
    return nn.Sequential(
        conv_layer(in_planes, base_planes, kernel_size=3, stride=2, padding=1, bias=False),
        norm_layer(base_planes),
        act_layer(inplace=True),
        nn.MaxPool2d(3, stride=2, padding=1)
    )


class ShuffleNetV1Backbone(nn.Module, ABC):

    def __init__(self,
                 in_channels=3,
                 base_channels=24,
                 groups=8,
                 stage_channels=(384, 768, 1536),
                 stage_blocks=(4, 8, 4),
                 downsamples=(1, 1, 1),
                 with_groups=(0, 1, 1),
                 block_layer=None,
                 conv_layer=None,
                 norm_layer=None,
                 act_layer=None,
                 ):
        """
        :param in_channels: 输入通道数
        :param base_channels: 第一个卷积层通道数
        :param groups: 分组数
        :param stage_channels: 每个阶段通道数
        :param stage_blocks: 每个阶段块个数
        :param downsamples: 是否执行空间下采样
        :param with_groups: 是否对第一个1x1逐点卷积执行分组操作
        :param block_layer: 块类型
        :param conv_layer: 卷积层类型
        :param norm_layer: 归一化层类型
        :param act_layer: 激活层类型
        """
        super(ShuffleNetV1Backbone, self).__init__()

        if block_layer is None:
            block_layer = ShuffleNetV1Unit
        if conv_layer is None:
            conv_layer = nn.Conv2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU

        self.stem = make_stem(in_channels,
                              base_channels,
                              conv_layer,
                              norm_layer,
                              act_layer
                              )

        in_channels = base_channels
        for i in range(len(stage_blocks)):
            res_layer = make_stage(in_channels,
                                   stage_channels[i],
                                   groups,
                                   stage_blocks[i],
                                   downsamples[i],
                                   with_groups[i],
                                   block_layer,
                                   conv_layer,
                                   norm_layer,
                                   act_layer
                                   )
            in_channels = stage_channels[i]
            stage_name = f'stage{i + 1}'
            self.add_module(stage_name, res_layer)
        self.stage_num = len(stage_blocks)

        init_weights(self.modules())

    def forward(self, x):
        x = self.stem(x)

        for i in range(self.stage_num):
            stage = self.__getattr__(f'stage{i + 1}')
            x = stage(x)

        return x


@registry.Backbone.register('ShuffleNetV1')
def build_sfv1_backbone(cfg):
    arch = cfg.MODEL.BACKBONE.ARCH
    in_channels = cfg.MODEL.BACKBONE.IN_PLANES
    downsamples = cfg.MODEL.BACKBONE.DOWNSAMPLES
    with_groups = cfg.MODEL.BACKBONE.WITH_GROUPS
    conv_layer = get_conv(cfg)
    norm_layer = get_norm(cfg)
    act_layer = get_act(cfg)

    block_layer, groups, base_channels, stage_channels, layer_blocks, width_multiplier = copy.deepcopy(
        arch_settings[arch])

    for i in range(len(stage_channels)):
        stage_channels[i] = int(stage_channels[i] * width_multiplier)

    return ShuffleNetV1Backbone(
        in_channels=in_channels,
        base_channels=base_channels,
        groups=groups,
        stage_channels=stage_channels,
        stage_blocks=layer_blocks,
        downsamples=downsamples,
        with_groups=with_groups,
        block_layer=block_layer,
        conv_layer=conv_layer,
        norm_layer=norm_layer,
        act_layer=act_layer,
    )

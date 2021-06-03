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
from .shufflenetv2_unit import ShuffleNetV2Unit

"""
some interesting issues:
* [通道设计不是2^n的原因 #47](https://github.com/megvii-model/ShuffleNet-Series/issues/47)
* [Compare time cost between mobilenetV1 and shufflenetV1、V2 #25](https://github.com/megvii-model/ShuffleNet-Series/issues/25)
* [Compare the speed of mobilenetv3-large, shufflenetv2 1.5x, shufflenetv2 + medium? #31](https://github.com/megvii-model/ShuffleNet-Series/issues/31)
* [question on why 4 fragments in parallel runs slower than 4 fragments in series #46](https://github.com/megvii-model/ShuffleNet-Series/issues/46)
"""

arch_settings = {
    'shufflenet_v2_x2_0': (ShuffleNetV2Unit, [244, 488, 976], (4, 8, 4), 2048),
    'shufflenet_v2_x1_5': (ShuffleNetV2Unit, [176, 352, 704], (4, 8, 4), 1024),
    'shufflenet_v2_x1_0': (ShuffleNetV2Unit, [116, 232, 464], (4, 8, 4), 1024),
    'shufflenet_v2_x0_5': (ShuffleNetV2Unit, [48, 96, 192], (4, 8, 4), 1024),
}


def make_stage(in_channels,
               out_channels,
               block_num,
               with_downsample,
               block_layer,
               conv_layer,
               norm_layer,
               act_layer,
               ):
    """
    :param in_channels: 输入通道数
    :param out_channels: 输出通道数
    :param block_num: 块个数
    :param with_downsample: 是否执行空间下采样
    :param block_layer: 块类型
    :param conv_layer: 卷积层类型
    :param norm_layer: 归一化层类型
    :param act_layer: 激活层类型
    :return:
    """
    stride = 2 if with_downsample else 1
    if with_downsample:
        down_sample = nn.Sequential(
            conv_layer(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, bias=False,
                       groups=in_channels),
            norm_layer(in_channels),
            conv_layer(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0, bias=False),
            norm_layer(out_channels // 2),
            act_layer(inplace=True)
        )
    else:
        down_sample = None

    blocks = list()
    blocks.append(block_layer(
        in_channels, out_channels, stride, down_sample, conv_layer, norm_layer, act_layer))
    in_channels = out_channels

    stride = 1
    down_sample = None
    for i in range(1, block_num):
        blocks.append(block_layer(
            in_channels // 2, out_channels, stride, down_sample, conv_layer, norm_layer, act_layer))
    return nn.Sequential(*blocks)


def make_stem(conv1_in_channels,
              conv1_out_channels,
              conv5_in_channels,
              conv5_out_channels,
              conv_layer,
              norm_layer,
              act_layer
              ):
    first_stem = nn.Sequential(
        conv_layer(conv1_in_channels, conv1_out_channels, kernel_size=3, stride=2, padding=1, bias=False),
        norm_layer(conv1_out_channels),
        act_layer(inplace=True),
        nn.MaxPool2d(3, stride=2, padding=1)
    )

    last_stem = nn.Sequential(
        conv_layer(conv5_in_channels, conv5_out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        norm_layer(conv5_out_channels),
        act_layer(inplace=True)
    )

    return first_stem, last_stem


class ShuffleNetV2Backbone(nn.Module, ABC):

    def __init__(self,
                 in_channels=3,
                 base_channels=24,
                 out_channels=1024,
                 stage_channels=(116, 232, 464),
                 stage_blocks=(4, 8, 4),
                 downsamples=(1, 1, 1),
                 block_layer=None,
                 conv_layer=None,
                 norm_layer=None,
                 act_layer=None
                 ):
        """
        :param in_channels: 输入通道数
        :param base_channels: 第一个卷积层通道数
        :param out_channels: 输出通道数
        :param stage_channels: 每一层通道数
        :param stage_blocks: 每一层块个数
        :param downsamples: 是否执行空间下采样
        :param block_layer: 块类型
        :param conv_layer: 卷积层类型
        :param norm_layer: 归一化层类型
        :param act_layer: 激活层类型
        """
        super(ShuffleNetV2Backbone, self).__init__()

        if block_layer is None:
            block_layer = ShuffleNetV2Unit
        if conv_layer is None:
            conv_layer = nn.Conv2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU

        self.first_stem, self.last_stem = make_stem(in_channels,
                                                    base_channels,
                                                    stage_channels[-1],
                                                    out_channels,
                                                    conv_layer,
                                                    norm_layer,
                                                    act_layer
                                                    )

        in_channels = base_channels
        for i in range(len(stage_blocks)):
            res_layer = make_stage(in_channels,
                                   stage_channels[i],
                                   stage_blocks[i],
                                   downsamples[i],
                                   block_layer,
                                   conv_layer,
                                   norm_layer,
                                   act_layer
                                   )
            in_channels = stage_channels[i]
            layer_name = f'stage{i + 1}'
            self.add_module(layer_name, res_layer)
        self.stage_num = len(stage_blocks)

        init_weights(self.modules())

    def forward(self, x):
        x = self.first_stem(x)

        for i in range(self.stage_num):
            stage = self.__getattr__(f'stage{i + 1}')
            x = stage(x)

        x = self.last_stem(x)
        return x


@registry.Backbone.register('ShuffleNetV2')
def build_sfv2_backbone(cfg):
    arch = cfg.MODEL.BACKBONE.ARCH
    in_channels = cfg.MODEL.BACKBONE.IN_PLANES
    base_channels = cfg.MODEL.BACKBONE.BASE_PLANES
    round_nearest = cfg.MODEL.COMPRESSION.ROUND_NEAREST

    block_layer, stage_channels, stage_blocks, out_channels = copy.deepcopy(arch_settings[arch])

    # base_channels = make_divisible(base_channels, round_nearest)
    # for i in range(len(stage_channels)):
    #     stage_channels[i] = make_divisible(stage_channels[i], round_nearest)
    # out_channels = make_divisible(out_channels, round_nearest)

    down_samples = cfg.MODEL.BACKBONE.DOWNSAMPLES
    conv_layer = get_conv(cfg)
    norm_layer = get_norm(cfg)
    act_layer = get_act(cfg)
    return ShuffleNetV2Backbone(
        in_channels=in_channels,
        base_channels=base_channels,
        out_channels=out_channels,
        stage_channels=stage_channels,
        stage_blocks=stage_blocks,
        downsamples=down_samples,
        block_layer=block_layer,
        conv_layer=conv_layer,
        norm_layer=norm_layer,
        act_layer=act_layer
    )

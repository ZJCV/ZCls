# -*- coding: utf-8 -*-

"""
@date: 2020/12/30 下午4:28
@file: mnasnet_unit.py
@author: zj
@description:
"""

from abc import ABC
import copy
import torch.nn as nn

from zcls.model import registry
from zcls.model.conv_helper import get_conv
from zcls.model.norm_helper import get_norm
from zcls.model.act_helper import get_act
from ..misc import round_to_multiple_of
from .mnasnet_unit import MNASNetUint, BN_MOMENTUM

arch_settings = {
    'mnasnet_a1': [
        # kernel_size, stride, expansion_rate, with_attention_2, repeated, out_planes
        [3, 1, 1, 0, 1, 16],
        [3, 2, 6, 0, 2, 24],
        [5, 2, 3, 1, 3, 40],
        [3, 2, 6, 0, 4, 80],
        [3, 1, 6, 1, 2, 112],
        [5, 2, 6, 1, 3, 160],
        [3, 1, 6, 0, 1, 320],
    ],
    'mnasnet_b1': [
        # refer to [vision/torchvision/models/mnasnet.py](https://github.com/pytorch/vision/blob/master/torchvision/models/mnasnet.py)
        # kernel_size, stride, expansion_rate, with_attention_2, repeated, out_planes
        [3, 1, 1, 0, 1, 16],
        [3, 2, 3, 0, 3, 24],
        [5, 2, 3, 0, 3, 40],
        [5, 2, 6, 0, 3, 80],
        [3, 1, 6, 0, 2, 96],
        [5, 2, 6, 0, 4, 192],
        [3, 1, 6, 0, 1, 320],
    ]
}


def make_stem(in_channels,
              base_channels,
              inner_channels,
              out_channels,
              conv_layer,
              norm_layer,
              act_layer):
    first_stem = nn.Sequential(
        conv_layer(in_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
        norm_layer(base_channels, momentum=BN_MOMENTUM),
        act_layer(inplace=True)
    )

    last_stem = nn.Sequential(
        conv_layer(inner_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        norm_layer(out_channels, momentum=BN_MOMENTUM),
        act_layer(inplace=True)
    )

    return first_stem, last_stem


class MNASNetBackbone(nn.Module, ABC):

    def __init__(self,
                 in_channels=3,
                 out_channels=1280,
                 stage_setting=None,
                 width_multiplier=1.,
                 round_nearest=8,
                 with_attention=True,
                 reduction=4,
                 attention_type='SqueezeAndExcitationBlock2D',
                 conv_layer=None,
                 norm_layer=None,
                 act_layer=None
                 ):
        """
        MnasNet-A1实现
        参考Torchvision mnasnet实现，在Backbone之后添加一个Conv1x1-BN-ReLU，最终输出特征维度为1280
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param stage_setting: 参数设置
        :param width_multiplier: 宽度乘法器
        :param round_nearest: 设置每一层通道数均为8的倍数
        :param with_attention: 是否使用注意力模块
        :param reduction: 衰减率
        :param attention_type: 注意力模块类型
        :param conv_layer: 卷积层类型
        :param norm_layer: 归一化层类型
        :param act_layer: 激活层类型
        """
        super(MNASNetBackbone, self).__init__()

        if conv_layer is None:
            conv_layer = nn.Conv2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU
        block_layer = MNASNetUint

        base_planes = 32
        if stage_setting is None:
            stage_setting = [
                # kernel_size, stride, expansion_rate, with_attention_2, repeated, out_planes
                [3, 1, 1, 0, 1, 16],
                [3, 2, 6, 0, 2, 24],
                [5, 2, 3, 1, 3, 40],
                [3, 2, 6, 0, 4, 80],
                [3, 1, 6, 1, 2, 112],
                [5, 2, 6, 1, 3, 160],
                [3, 1, 6, 0, 1, 320],
            ]

        base_planes = round_to_multiple_of(base_planes * width_multiplier, round_nearest)
        # 参考Torchvision实现，不对输出特征维度进行缩放
        # out_planes = round_to_multiple_of(out_planes * width_multiplier, round_nearest)
        for i in range(len(stage_setting)):
            stage_setting[i][-1] = round_to_multiple_of(stage_setting[i][-1] * width_multiplier, round_nearest)

        self.first_stem, self.last_stem = make_stem(in_channels,
                                                    base_planes,
                                                    stage_setting[-1][-1],
                                                    out_channels,
                                                    conv_layer,
                                                    norm_layer,
                                                    act_layer)

        in_channels = base_planes
        for i, (kernel_size, stride, expansion_rate, with_attention_2, repeated, out_channels) in enumerate(
                stage_setting):
            features = list()
            for j in range(repeated):
                stride = stride if j == 0 else 1
                features.append(block_layer(in_channels,
                                            out_channels,
                                            stride=stride,
                                            kernel_size=kernel_size,
                                            expansion_rate=expansion_rate,
                                            with_attention=with_attention and with_attention_2,
                                            reduction=reduction,
                                            attention_type=attention_type,
                                            conv_layer=conv_layer,
                                            norm_layer=norm_layer,
                                            act_layer=act_layer
                                            ))
                in_channels = out_channels
            stage_name = f'stage{i + 1}'
            self.add_module(stage_name, nn.Sequential(*features))
        self.stage_num = len(stage_setting)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="sigmoid")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.first_stem(x)

        for i in range(self.stage_num):
            stage = self.__getattr__(f'stage{i + 1}')
            x = stage(x)

        x = self.last_stem(x)
        return x


@registry.Backbone.register('MNASNet')
def build_mnasnet(cfg):
    arch = cfg.MODEL.BACKBONE.ARCH
    in_planes = cfg.MODEL.BACKBONE.IN_PLANES
    norm_layer = get_norm(cfg)
    # compression
    width_multiplier = cfg.MODEL.COMPRESSION.WIDTH_MULTIPLIER
    round_nearest = cfg.MODEL.COMPRESSION.ROUND_NEAREST
    # attention
    with_attention = cfg.MODEL.ATTENTION.WITH_ATTENTION
    reduction = cfg.MODEL.ATTENTION.REDUCTION
    attention_type = cfg.MODEL.ATTENTION.ATTENTION_TYPE
    # conv
    conv_layer = get_conv(cfg)
    # act
    act_layer = get_act(cfg)

    out_planes = 1280
    stage_setting = copy.deepcopy(arch_settings[arch])
    return MNASNetBackbone(
        in_channels=in_planes,
        out_channels=out_planes,
        stage_setting=stage_setting,
        width_multiplier=width_multiplier,
        round_nearest=round_nearest,
        with_attention=with_attention,
        reduction=reduction,
        attention_type=attention_type,
        conv_layer=conv_layer,
        norm_layer=norm_layer,
        act_layer=act_layer
    )

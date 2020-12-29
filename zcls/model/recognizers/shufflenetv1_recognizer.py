# -*- coding: utf-8 -*-

"""
@date: 2020/12/24 下午7:38
@file: shufflenetv1_recognizer.py
@author: zj
@description: 
"""
from abc import ABC

import torch.nn as nn
from torch.nn.modules.module import T
from torchvision.models.utils import load_state_dict_from_url

from zcls.config.key_word import KEY_OUTPUT
from .. import registry
from ..backbones.shufflenetv1_unit import ShuffleNetV1Unit
from ..backbones.shufflenetv1_backbone import ShuffleNetV1Backbone
from ..heads.shufflenetv1_head import ShuffleNetV1Head
from ..norm_helper import get_norm, freezing_bn
from ..conv_helper import get_conv
from ..act_helper import get_act

"""
Note 1: Empirically g = 3 usually has a proper trade-off between accuracy and actual inference time
Note 2: Comparing ShuffleNet 2× with MobileNet whose complexity are comparable (524 vs. 569 MFLOPs)
"""

arch_settings = {
    'shufflenetv1_1g1x': (ShuffleNetV1Unit, 1, (144, 288, 576), (4, 8, 4), 1.0),
    'shufflenetv1_1g0.5x': (ShuffleNetV1Unit, 1, (144, 288, 576), (4, 8, 4), 0.5),
    'shufflenetv1_1g0.25x': (ShuffleNetV1Unit, 1, (144, 288, 576), (4, 8, 4), 0.25),
    'shufflenetv1_2g1x': (ShuffleNetV1Unit, 2, (200, 400, 800), (4, 8, 4), 1.0),
    'shufflenetv1_2g0.5x': (ShuffleNetV1Unit, 2, (200, 400, 800), (4, 8, 4), 0.5),
    'shufflenetv1_2g0.25x': (ShuffleNetV1Unit, 2, (200, 400, 800), (4, 8, 4), 0.25),
    'shufflenetv1_3g2x': (ShuffleNetV1Unit, 3, (240, 480, 960), (4, 8, 4), 2.0),
    'shufflenetv1_3g1x': (ShuffleNetV1Unit, 3, (240, 480, 960), (4, 8, 4), 1.0),
    'shufflenetv1_3g0.5x': (ShuffleNetV1Unit, 3, (240, 480, 960), (4, 8, 4), 0.5),
    'shufflenetv1_3g0.25x': (ShuffleNetV1Unit, 3, (240, 480, 960), (4, 8, 4), 0.25),
    'shufflenetv1_4g1x': (ShuffleNetV1Unit, 4, (272, 544, 1088), (4, 8, 4), 1.0),
    'shufflenetv1_4g0.5x': (ShuffleNetV1Unit, 4, (272, 544, 1088), (4, 8, 4), 0.5),
    'shufflenetv1_4g0.25x': (ShuffleNetV1Unit, 4, (272, 544, 1088), (4, 8, 4), 0.25),
    'shufflenetv1_8g1x': (ShuffleNetV1Unit, 8, (384, 768, 1536), (4, 8, 4), 1.0),
    'shufflenetv1_8g0.5x': (ShuffleNetV1Unit, 8, (384, 768, 1536), (4, 8, 4), 0.5),
    'shufflenetv1_8g0.25x': (ShuffleNetV1Unit, 8, (384, 768, 1536), (4, 8, 4), 0.25),
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ShuffleNetV1Recognizer(nn.Module, ABC):

    def __init__(self,
                 ######################### for recognizer
                 # 架构
                 arch='shufflenetv1_1g1x',
                 # 设置每一层通道数均为8的倍数
                 round_nearest=8,
                 # ZCls预训练模型
                 zcls_pretrained="",
                 # 假定预训练模型类别数
                 pretrained_num_classes=1000,
                 # 固定BN
                 fix_bn=False,
                 # 仅训练第一层BN
                 partial_bn=False,
                 ########################## for head
                 # 随机失火概率
                 dropout_rate=0.,
                 # 类别数
                 num_classes=1000,
                 ########################## for backbone
                 # 输入通道数
                 in_planes=3,
                 # 第一个卷积层通道数
                 base_planes=24,
                 # 是否执行空间下采样
                 down_samples=(1, 1, 1),
                 # 是否对第一个1x1逐点卷积执行分组操作
                 with_groups=(0, 1, 1),
                 # 卷积层类型
                 conv_layer=None,
                 # 归一化层类型
                 norm_layer=None,
                 # 激活层类型
                 act_layer=None,
                 ):
        super(ShuffleNetV1Recognizer, self).__init__()
        self.fix_bn = fix_bn
        self.partial_bn = partial_bn

        block_layer, groups, layer_planes, layer_blocks, width_multiplier = arch_settings[arch]

        base_planes = _make_divisible(base_planes * width_multiplier, round_nearest)
        stage_planes = list()
        for i in range(len(layer_planes)):
            stage_planes.append(_make_divisible(layer_planes[i] * width_multiplier, round_nearest))
        layer_planes = stage_planes

        self.backbone = ShuffleNetV1Backbone(
            # 输入通道数
            in_planes=in_planes,
            # 第一个卷积层通道数
            base_planes=base_planes,
            # 分组数
            groups=groups,
            # 每个阶段通道数
            layer_planes=layer_planes,
            # 每个阶段块个数
            layer_blocks=layer_blocks,
            # 是否执行空间下采样
            down_samples=down_samples,
            # 是否对第一个1x1逐点卷积执行分组操作
            with_groups=with_groups,
            # 块类型
            block_layer=block_layer,
            # 卷积层类型
            conv_layer=conv_layer,
            # 归一化层类型
            norm_layer=norm_layer,
            # 激活层类型
            act_layer=act_layer,
        )
        feature_dims = layer_planes[-1]
        self.head = ShuffleNetV1Head(
            feature_dims=feature_dims,
            num_classes=pretrained_num_classes
        )

        self.init_weights(zcls_pretrained, pretrained_num_classes, num_classes)

    def init_weights(self, pretrained, pretrained_num_classes, num_classes):
        if pretrained != "":
            state_dict = load_state_dict_from_url(pretrained, progress=True)
            self.load_state_dict(state_dict=state_dict, strict=False)
        if num_classes != pretrained_num_classes:
            fc = self.head.fc
            fc_features = fc.in_features
            self.head.fc = nn.Linear(fc_features, num_classes)
            self.head.init_weights()

    def train(self, mode: bool = True) -> T:
        super(ShuffleNetV1Recognizer, self).train(mode=mode)

        if mode and (self.partial_bn or self.fix_bn):
            freezing_bn(self, partial_bn=self.partial_bn)

        return self

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)

        return {KEY_OUTPUT: x}


@registry.RECOGNIZER.register('ShuffleNetV1')
def build_shufflenet_v1(cfg):
    # for recognizer
    arch = cfg.MODEL.BACKBONE.ARCH
    pretrained = cfg.MODEL.RECOGNIZER.PRETRAINED
    pretrained_num_classes = cfg.MODEL.RECOGNIZER.PRETRAINED_NUM_CLASSES
    fix_bn = cfg.MODEL.NORM.FIX_BN
    partial_bn = cfg.MODEL.NORM.PARTIAL_BN
    # for head
    dropout_rate = cfg.MODEL.HEAD.DROPOUT
    num_classes = cfg.MODEL.HEAD.NUM_CLASSES
    # compression
    round_nearest = cfg.MODEL.COMPRESSION.ROUND_NEAREST
    # for backbone
    in_planes = cfg.MODEL.BACKBONE.IN_PLANES
    base_planes = cfg.MODEL.BACKBONE.BASE_PLANES
    down_samples = cfg.MODEL.BACKBONE.DOWN_SAMPLES
    with_groups = cfg.MODEL.BACKBONE.WITH_GROUPS
    # other
    conv_layer = get_conv(cfg)
    norm_layer = get_norm(cfg)
    act_layer = get_act(cfg)

    return ShuffleNetV1Recognizer(
        ######################### for recognizer
        # 架构
        arch=arch,
        # 设置每一层通道数均为8的倍数
        round_nearest=round_nearest,
        # ZCls预训练模型
        zcls_pretrained=pretrained,
        # 假定预训练模型类别数
        pretrained_num_classes=pretrained_num_classes,
        # 固定BN
        fix_bn=fix_bn,
        # 仅训练第一层BN
        partial_bn=partial_bn,
        ########################## for head
        # 随机失火概率
        dropout_rate=dropout_rate,
        # 类别数
        num_classes=num_classes,
        ########################## for backbone
        # 输入通道数
        in_planes=in_planes,
        # 第一个卷积层通道数
        base_planes=base_planes,
        # 是否执行空间下采样
        down_samples=down_samples,
        # 是否对第一个1x1逐点卷积执行分组操作
        with_groups=with_groups,
        # 卷积层类型
        conv_layer=conv_layer,
        # 归一化层类型
        norm_layer=norm_layer,
        # 激活层类型
        act_layer=act_layer,
    )

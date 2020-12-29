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

from .. import registry
from ..backbones.shufflenetv1_backbone import ShuffleNetV1Backbone
from ..heads.shufflenetv1_head import ShuffleNetV1Head
from ..norm_helper import get_norm, freezing_bn

"""
Note 1: Empirically g = 3 usually has a proper trade-off between accuracy and actual inference time
Note 2: Comparing ShuffleNet 2× with MobileNet whose complexity are comparable (524 vs. 569 MFLOPs)
"""

arch_settings = {
    'shufflenetv1_1g1x': (1, (144, 288, 576), 1.0),
    'shufflenetv1_1g0.5x': (1, (144, 288, 576), 0.5),
    'shufflenetv1_1g0.25x': (1, (144, 288, 576), 0.25),
    'shufflenetv1_2g1x': (2, (200, 400, 800), 1.0),
    'shufflenetv1_2g0.5x': (2, (200, 400, 800), 0.5),
    'shufflenetv1_2g0.25x': (2, (200, 400, 800), 0.25),
    'shufflenetv1_3g2x': (3, (240, 480, 960), 2.0),
    'shufflenetv1_3g1x': (3, (240, 480, 960), 1.0),
    'shufflenetv1_3g0.5x': (3, (240, 480, 960), 0.5),
    'shufflenetv1_3g0.25x': (3, (240, 480, 960), 0.25),
    'shufflenetv1_4g1x': (4, (272, 544, 1088), 1.0),
    'shufflenetv1_4g0.5x': (4, (272, 544, 1088), 0.5),
    'shufflenetv1_4g0.25x': (4, (272, 544, 1088), 0.25),
    'shufflenetv1_8g1x': (8, (384, 768, 1536), 1.0),
    'shufflenetv1_8g0.5x': (8, (384, 768, 1536), 0.5),
    'shufflenetv1_8g0.25x': (8, (384, 768, 1536), 0.25),
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
                 # 输入通道数
                 inplanes=3,
                 # 类别数
                 num_classes=1000,
                 # 第一个卷积层通道数
                 base_channel=24,
                 # 分组数
                 groups=8,
                 # 每一层通道数
                 layer_planes=(384, 768, 1536),
                 # 每一层块个数
                 layer_blocks=(4, 8, 4),
                 # 是否对第一个1x1逐点卷积执行分组操作
                 with_groups=(0, 1, 1),
                 # 宽度乘法器
                 width_multiplier=1.0,
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
                 # 卷积层类型
                 conv_layer=None,
                 # 归一化层类型
                 norm_layer=None,
                 # 激活层类型
                 act_layer=None
                 ):
        super(ShuffleNetV1Recognizer, self).__init__()
        self.num_classes = num_classes
        self.fix_bn = fix_bn
        self.partial_bn = partial_bn

        base_channel = _make_divisible(base_channel * width_multiplier, round_nearest)
        for i in range(len(layer_planes)):
            layer_planes[i] = _make_divisible(layer_planes[i] * width_multiplier, round_nearest)
        feature_dims = layer_planes[-1]

        self.backbone = ShuffleNetV1Backbone(
            inplanes=inplanes,
            base_channel=base_channel,
            groups=groups,
            layer_planes=layer_planes,
            layer_blocks=layer_blocks,
            with_groups=with_groups,
            conv_layer=conv_layer,
            norm_layer=norm_layer,
            act_layer=act_layer
        )
        self.head = ShuffleNetV1Head(
            feature_dims=feature_dims,
            num_classes=pretrained_num_classes
        )

        self.init_weights(pretrained=zcls_pretrained, pretrained_num_classes=pretrained_num_classes)

    def init_weights(self, pretrained="", pretrained_num_classes=1000):
        if pretrained != "":
            state_dict = load_state_dict_from_url(pretrained, progress=True)
            self.load_state_dict(state_dict=state_dict, strict=False)
        if self.num_classes != pretrained_num_classes:
            fc = self.head.fc
            fc_features = fc.in_features
            self.head.fc = nn.Linear(fc_features, self.num_classes)

            nn.init.normal_(self.head.fc.weight, 0, 0.01)
            nn.init.zeros_(self.head.fc.bias)

    def train(self, mode: bool = True) -> T:
        super(ShuffleNetV1Recognizer, self).train(mode=mode)

        if mode and (self.partial_bn or self.fix_bn):
            freezing_bn(self, partial_bn=self.partial_bn)

        return self

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)

        return {'probs': x}


@registry.RECOGNIZER.register('ShuffleNetV1')
def build_resnet(cfg):
    type = cfg.MODEL.RECOGNIZER.NAME
    zcls_pretrained = cfg.MODEL.ZClS_PRETRAINED
    pretrained_num_classes = cfg.MODEL.PRETRAINED_NUM_CLASSES
    arch = cfg.MODEL.BACKBONE.ARCH
    num_classes = cfg.MODEL.HEAD.NUM_CLASSES
    fix_bn = cfg.MODEL.NORM.FIX_BN
    partial_bn = cfg.MODEL.NORM.PARTIAL_BN

    norm_layer = get_norm(cfg)

    groups, layer_planes, width_multiplier = arch_settings[arch]

    return ShuffleNetV1Recognizer(
        num_classes=num_classes,
        groups=groups,
        layer_planes=layer_planes,
        width_multiplier=width_multiplier,
        zcls_pretrained=zcls_pretrained,
        pretrained_num_classes=pretrained_num_classes,
        fix_bn=fix_bn,
        partial_bn=partial_bn,
        norm_layer=norm_layer
    )

# -*- coding: utf-8 -*-

"""
@date: 2020/12/2 下午9:38
@file: mobilenetv1_recognizer.py
@author: zj
@description: 
"""
from abc import ABC

import torch.nn as nn
from torchvision.models import mobilenet_v2

from .. import registry
from ..backbones.mobilenetv2_backbone import MobileNetV2Backbone
from ..heads.mobilenetv2_head import MobileNetV2Head
from ..norm_helper import get_norm, freezing_bn
from ..act_helper import get_act


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


class MobileNetV2Recognizer(nn.Module, ABC):

    def __init__(self,
                 # 输入通道数
                 inplanes=3,
                 # 第一个卷积层通道数
                 base_channel=32,
                 # Head输入特征数
                 feature_dims=1280,
                 # 类别数
                 num_classes=1000,
                 # 宽度乘法器
                 width_multiplier=1.0,
                 # 设置每一层通道数均为8的倍数
                 round_nearest=8,
                 # 固定BN
                 fix_bn=False,
                 # 仅训练第一层BN
                 partial_bn=False,
                 # 反向残差块设置
                 inverted_residual_setting=None,
                 # 卷积层类型
                 conv_layer=None,
                 # 归一化层类型
                 norm_layer=None,
                 # 激活层类型
                 act_layer=None
                 ):
        super(MobileNetV2Recognizer, self).__init__()
        self.num_classes = num_classes
        self.fix_bn = fix_bn
        self.partial_bn = partial_bn

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
        else:
            assert len(inverted_residual_setting[0]) == 4

        base_channel = _make_divisible(base_channel * width_multiplier, round_nearest)
        for i in range(len(inverted_residual_setting)):
            channel = inverted_residual_setting[i][1]
            inverted_residual_setting[i][1] = _make_divisible(channel * width_multiplier, round_nearest)
        feature_dims = _make_divisible(feature_dims * width_multiplier, round_nearest)

        self.backbone = MobileNetV2Backbone(
            inplanes=inplanes,
            planes=feature_dims,
            base_channel=base_channel,
            inverted_residual_setting=inverted_residual_setting,
            conv_layer=conv_layer,
            norm_layer=norm_layer,
            act_layer=act_layer
        )
        self.head = MobileNetV2Head(
            feature_dims=feature_dims,
            num_classes=1000
        )

    def train(self, mode: bool = True):
        super(MobileNetV2Recognizer, self).train(mode=mode)

        if mode and (self.partial_bn or self.fix_bn):
            freezing_bn(self, partial_bn=self.partial_bn)

        return self

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)

        return {'probs': x}


class MobileNetV2_Pytorch(nn.Module):

    def __init__(self,
                 num_classes=1000,
                 width_multiplier=1.0,
                 torchvision_pretrained=False,
                 fix_bn=False,
                 partial_bn=False,
                 norm_layer=None):
        super(MobileNetV2_Pytorch, self).__init__()

        self.num_classes = num_classes
        self.fix_bn = fix_bn
        self.partial_bn = partial_bn

        self.model = mobilenet_v2(pretrained=torchvision_pretrained,
                                  width_mult=width_multiplier,
                                  norm_layer=norm_layer)

        self._init_weights()

    def _init_weights(self):
        if self.num_classes != 1000:
            fc = self.model.classifier[1]
            fc_features = fc.in_features
            fc = nn.Linear(fc_features, self.num_classes)

            nn.init.normal_(fc.weight, 0, 0.01)
            nn.init.zeros_(fc.bias)

            self.model.classifier[1] = fc

    def train(self, mode: bool = True):
        super(MobileNetV2_Pytorch, self).train(mode=mode)

        if mode and (self.partial_bn or self.fix_bn):
            freezing_bn(self, partial_bn=self.partial_bn)

        return self

    def forward(self, x):
        x = self.model(x)

        return {'probs': x}


@registry.RECOGNIZER.register('MobileNetV2')
def build_mobilenetv2(cfg):
    type = cfg.MODEL.RECOGNIZER.NAME
    torchvision_pretrained = cfg.MODEL.TORCHVISION_PRETRAINED
    num_classes = cfg.MODEL.HEAD.NUM_CLASSES
    fix_bn = cfg.MODEL.NORM.FIX_BN
    partial_bn = cfg.MODEL.NORM.PARTIAL_BN

    norm_layer = get_norm(cfg)
    width_multiplier = cfg.MODEL.COMPRESSION.WIDTH_MULTIPLIER

    if type == 'MobileNetV2_Pytorch':
        return MobileNetV2_Pytorch(
            num_classes=num_classes,
            torchvision_pretrained=torchvision_pretrained,
            width_multiplier=width_multiplier,
            fix_bn=fix_bn,
            partial_bn=partial_bn,
            norm_layer=norm_layer
        )
    elif type == 'MobileNetV2_Custom':
        feature_dims = cfg.MODEL.HEAD.FEATURE_DIMS
        act_layer = get_act(cfg)

        return MobileNetV2Recognizer(
            feature_dims=feature_dims,
            num_classes=num_classes,
            width_multiplier=width_multiplier,
            fix_bn=fix_bn,
            partial_bn=partial_bn,
            norm_layer=norm_layer,
            act_layer=act_layer
        )
    else:
        raise ValueError(f'{type} does not exist')

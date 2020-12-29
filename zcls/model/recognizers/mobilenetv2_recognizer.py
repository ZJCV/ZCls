# -*- coding: utf-8 -*-

"""
@date: 2020/12/2 下午9:38
@file: mobilenetv1_recognizer.py
@author: zj
@description: 
"""
from abc import ABC

import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models import mobilenet_v2

from zcls.config.key_word import KEY_OUTPUT
from .. import registry
from ..backbones.mobilenetv2_backbone import MobileNetV2Backbone
from ..heads.mobilenetv2_head import MobileNetV2Head
from ..norm_helper import get_norm, freezing_bn
from ..act_helper import get_act
from ..conv_helper import get_conv


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
                 in_planes=3,
                 # 第一个卷积层通道数
                 base_planes=32,
                 # Head输入特征数
                 out_planes=1280,
                 # 随机失活概率
                 dropout_rate=0.,
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
                 # zcls预训练模型
                 pretrained="",
                 # 预训练模型类别数
                 pretrained_num_classes=1000,
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

        base_planes = _make_divisible(base_planes * width_multiplier, round_nearest)
        for i in range(len(inverted_residual_setting)):
            channel = inverted_residual_setting[i][1]
            inverted_residual_setting[i][1] = _make_divisible(channel * width_multiplier, round_nearest)
        out_planes = _make_divisible(out_planes * width_multiplier, round_nearest)

        self.backbone = MobileNetV2Backbone(
            in_planes=in_planes,
            out_planes=out_planes,
            base_planes=base_planes,
            inverted_residual_setting=inverted_residual_setting,
            conv_layer=conv_layer,
            norm_layer=norm_layer,
            act_layer=act_layer
        )
        self.head = MobileNetV2Head(
            feature_dims=out_planes,
            dropout_rate=dropout_rate,
            num_classes=pretrained_num_classes
        )

        self.init_weights(pretrained=pretrained,
                          pretrained_num_classes=pretrained_num_classes,
                          num_classes=num_classes)

    def init_weights(self,
                     pretrained,
                     pretrained_num_classes,
                     num_classes
                     ):
        if pretrained != "":
            state_dict = load_state_dict_from_url(pretrained, progress=True)
            self.backbone.load_state_dict(state_dict, strict=False)
            self.head.load_state_dict(state_dict, strict=False)
        if num_classes != pretrained_num_classes:
            fc = self.head.fc
            fc_features = fc.in_features
            self.head.fc = nn.Linear(fc_features, num_classes)

            nn.init.normal_(self.head.fc.weight, 0, 0.01)
            nn.init.zeros_(self.head.fc.bias)

    def train(self, mode: bool = True):
        super(MobileNetV2Recognizer, self).train(mode=mode)

        if mode and (self.partial_bn or self.fix_bn):
            freezing_bn(self, partial_bn=self.partial_bn)

        return self

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)

        return {KEY_OUTPUT: x}


class TorchvisionMobileNetV2(nn.Module, ABC):

    def __init__(self,
                 num_classes=1000,
                 width_multiplier=1.0,
                 torchvision_pretrained=False,
                 pretrained_num_classes=1000,
                 fix_bn=False,
                 partial_bn=False,
                 norm_layer=None):
        super(TorchvisionMobileNetV2, self).__init__()
        self.fix_bn = fix_bn
        self.partial_bn = partial_bn

        self.model = mobilenet_v2(pretrained=torchvision_pretrained,
                                  width_mult=width_multiplier,
                                  norm_layer=norm_layer,
                                  num_classes=pretrained_num_classes)

        self.init_weights(num_classes, pretrained_num_classes)

    def init_weights(self, num_classes, pretrained_num_classes):
        if num_classes != pretrained_num_classes:
            fc = self.model.classifier[1]
            fc_features = fc.in_features
            fc = nn.Linear(fc_features, num_classes)

            nn.init.normal_(fc.weight, 0, 0.01)
            nn.init.zeros_(fc.bias)

            self.model.classifier[1] = fc

    def train(self, mode: bool = True):
        super(TorchvisionMobileNetV2, self).train(mode=mode)

        if mode and (self.partial_bn or self.fix_bn):
            freezing_bn(self, partial_bn=self.partial_bn)

        return self

    def forward(self, x):
        x = self.model(x)

        return {KEY_OUTPUT: x}


@registry.RECOGNIZER.register('MobileNetV2')
def build_mobilenet_v2(cfg):
    # for recognizer
    recognizer_name = cfg.MODEL.RECOGNIZER.NAME
    torchvision_pretrained = cfg.MODEL.RECOGNIZER.TORCHVISION_PRETRAINED
    pretrained_num_classes = cfg.MODEL.RECOGNIZER.PRETRAINED_NUM_CLASSES
    fix_bn = cfg.MODEL.NORM.FIX_BN
    partial_bn = cfg.MODEL.NORM.PARTIAL_BN
    # for head
    num_classes = cfg.MODEL.HEAD.NUM_CLASSES
    # other
    norm_layer = get_norm(cfg)
    # for compression
    width_multiplier = cfg.MODEL.COMPRESSION.WIDTH_MULTIPLIER

    if recognizer_name == 'TorchvisionMobileNetV2':
        return TorchvisionMobileNetV2(
            num_classes=num_classes,
            torchvision_pretrained=torchvision_pretrained,
            pretrained_num_classes=pretrained_num_classes,
            width_multiplier=width_multiplier,
            fix_bn=fix_bn,
            partial_bn=partial_bn,
            norm_layer=norm_layer
        )
    elif recognizer_name == 'CustomMobileNetV2':
        # for backbone
        in_planes = cfg.MODEL.BACKBONE.IN_PLANES
        base_planes = cfg.MODEL.BACKBONE.BASE_PLANES
        feature_dims = cfg.MODEL.BACKBONE.FEATURE_DIMS
        # for head
        dropout_rate = cfg.MODEL.HEAD.DROPOUT
        # for compression
        round_nearest = cfg.MODEL.COMPRESSION.ROUND_NEAREST
        # for recognizer
        pretrained_num_classes = cfg.MODEL.RECOGNIZER.PRETRAINED_NUM_CLASSES
        pretrained = cfg.MODEL.RECOGNIZER.PRETRAINED
        # for other
        conv_layer = get_conv(cfg)
        act_layer = get_act(cfg)

        return MobileNetV2Recognizer(
            # 输入通道数
            in_planes=in_planes,
            # 第一个卷积层通道数
            base_planes=base_planes,
            # Head输入特征数
            out_planes=feature_dims,
            # 随机失活概率
            dropout_rate=dropout_rate,
            # 类别数
            num_classes=num_classes,
            # 宽度乘法器
            width_multiplier=width_multiplier,
            # 设置每一层通道数均为8的倍数
            round_nearest=round_nearest,
            # 固定BN
            fix_bn=fix_bn,
            # 仅训练第一层BN
            partial_bn=partial_bn,
            # zcls预训练模型
            pretrained=pretrained,
            # 预训练模型类别数
            pretrained_num_classes=pretrained_num_classes,
            # 卷积层类型
            conv_layer=conv_layer,
            # 归一化层类型
            norm_layer=norm_layer,
            # 激活层类型
            act_layer=act_layer
        )
    else:
        raise ValueError(f'{type} does not exist')

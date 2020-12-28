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

from zcls.config.key_word import KEY_OUTPUT
from .. import registry
from ..backbones.mobilenetv1_backbone import MobileNetV1Backbone
from ..heads.mobilenetv1_head import MobileNetV1Head
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


class MobileNetV1Recognizer(nn.Module, ABC):

    def __init__(self,
                 # 输入通道数
                 in_planes=3,
                 # 第一个卷积层通道数
                 base_planes=32,
                 # 后续各深度卷积通道数
                 layer_planes=(64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024),
                 # 卷积步长
                 strides=(1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 2),
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
                 # 卷积层类型
                 conv_layer=None,
                 # 归一化层类型
                 norm_layer=None,
                 # 激活层类型
                 act_layer=None
                 ):
        super(MobileNetV1Recognizer, self).__init__()
        self.num_classes = num_classes
        self.fix_bn = fix_bn
        self.partial_bn = partial_bn

        base_planes = _make_divisible(base_planes * width_multiplier, round_nearest)
        layer_planes = [_make_divisible(layer_plane * width_multiplier, round_nearest) for layer_plane in layer_planes]
        feature_dims = _make_divisible(layer_planes[-1], round_nearest)

        self.backbone = MobileNetV1Backbone(
            in_planes=in_planes,
            base_planes=base_planes,
            layer_planes=layer_planes,
            strides=strides,
            conv_layer=conv_layer,
            norm_layer=norm_layer,
            act_layer=act_layer
        )
        self.head = MobileNetV1Head(
            feature_dims=feature_dims,
            dropout_rate=dropout_rate,
            num_classes=pretrained_num_classes
        )

        self._init_weights(pretrained=pretrained,
                           pretrained_num_classes=pretrained_num_classes,
                           num_classes=num_classes)

    def _init_weights(self,
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
        super(MobileNetV1Recognizer, self).train(mode=mode)

        if mode and (self.partial_bn or self.fix_bn):
            freezing_bn(self, partial_bn=self.partial_bn)

        return self

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)

        return {KEY_OUTPUT: x}


@registry.RECOGNIZER.register('MobileNetV1')
def build_mobilenet_v1(cfg):
    # for recognizer
    pretrained_num_classes = cfg.MODEL.RECOGNIZER.PRETRAINED_NUM_CLASSES
    pretrained = cfg.MODEL.RECOGNIZER.PRETRAINED
    # for head
    dropout_rate = cfg.MODEL.HEAD.DROPOUT
    num_classes = cfg.MODEL.HEAD.NUM_CLASSES
    # for backbone
    in_planes = cfg.MODEL.BACKBONE.IN_PLANES
    base_planes = cfg.MODEL.BACKBONE.BASE_PLANES
    layer_planes = cfg.MODEL.BACKBONE.LAYER_PLANES
    strides = cfg.MODEL.BACKBONE.STRIDES
    # other
    conv_layer = get_conv(cfg)
    norm_layer = get_norm(cfg)
    act_layer = get_act(cfg)
    fix_bn = cfg.MODEL.NORM.FIX_BN
    partial_bn = cfg.MODEL.NORM.PARTIAL_BN
    width_multiplier = cfg.MODEL.COMPRESSION.WIDTH_MULTIPLIER
    round_nearest = cfg.MODEL.COMPRESSION.ROUND_NEAREST

    model = MobileNetV1Recognizer(
        # 输入通道数
        in_planes=in_planes,
        # 第一个卷积层通道数
        base_planes=base_planes,
        # 后续各深度卷积通道数
        layer_planes=layer_planes,
        # 卷积步长
        strides=strides,
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

    return model

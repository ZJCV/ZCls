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
from ..backbones.mobilenetv3_backbone import MobileNetV3Backbone
from ..heads.mobilenetv3_head import MobileNetV3Head
from ..norm_helper import get_norm, freezing_bn
from ..conv_helper import get_conv
from ..act_helper import get_act, get_sigmoid

arch_settings = {
    'mobilenetv3-large': [16, 960, 1280,
                          [
                              # kernel_size, stride, inner_planes, with_attention, non-linearity, out_planes
                              [3, 1, 16, 0, 'RE', 16],
                              [3, 2, 64, 0, 'RE', 24],
                              [3, 1, 72, 0, 'RE', 24],
                              [5, 2, 72, 1, 'RE', 40],
                              [5, 1, 120, 1, 'RE', 40],
                              [5, 1, 120, 1, 'RE', 40],
                              [3, 2, 240, 0, 'HS', 80],
                              [3, 1, 200, 0, 'HS', 80],
                              [3, 1, 184, 0, 'HS', 80],
                              [3, 1, 184, 0, 'HS', 80],
                              [3, 1, 480, 1, 'HS', 112],
                              [3, 1, 672, 1, 'HS', 112],
                              [5, 2, 672, 1, 'HS', 160],
                              [5, 1, 960, 1, 'HS', 160],
                              [5, 1, 960, 1, 'HS', 160],
                          ]],
    'mobilenetv3-small': [16, 576, 1024,
                          [
                              # kernel_size, stride, inner_planes, with_attention, non-linearity, out_planes
                              [3, 2, 16, 1, 'RE', 16],
                              [3, 2, 72, 0, 'RE', 24],
                              [3, 1, 88, 0, 'RE', 24],
                              [5, 2, 96, 1, 'HS', 40],
                              [5, 1, 240, 1, 'HS', 40],
                              [5, 1, 240, 1, 'HS', 40],
                              [5, 1, 120, 1, 'HS', 48],
                              [5, 1, 144, 1, 'HS', 48],
                              [5, 2, 288, 1, 'HS', 96],
                              [5, 1, 576, 1, 'HS', 96],
                              [5, 1, 576, 1, 'HS', 96]
                          ]]
}


class MobileNetV3Recognizer(nn.Module, ABC):

    def __init__(self,
                 # 架构
                 arch='mobilenetv3-large',
                 # 输入通道数
                 in_planes=3,
                 # 宽度乘法器
                 width_multiplier=1.,
                 # 设置每一层通道数均为8的倍数
                 round_nearest=8,
                 # 是否使用注意力模块
                 with_attention=True,
                 # 衰减率
                 reduction=4,
                 # 注意力模块类型
                 attention_type='SqueezeAndExcitationBlock2D',
                 # 类别数
                 num_classes=1000,
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
                 act_layer=None,
                 # sigmoid类型
                 sigmoid_type=None
                 ):
        super(MobileNetV3Recognizer, self).__init__()
        self.fix_bn = fix_bn
        self.partial_bn = partial_bn

        base_planes, feature_dims, inner_dims, layer_setting = arch_settings[arch]

        self.backbone = MobileNetV3Backbone(
            in_planes=in_planes,
            base_planes=base_planes,
            out_planes=feature_dims,
            width_multiplier=width_multiplier,
            round_nearest=round_nearest,
            with_attention=with_attention,
            reduction=reduction,
            attention_type=attention_type,
            conv_layer=conv_layer,
            norm_layer=norm_layer,
            act_layer=act_layer,
            sigmoid_type=sigmoid_type
        )
        self.head = MobileNetV3Head(
            feature_dims=feature_dims,
            inner_dims=inner_dims,
            num_classes=pretrained_num_classes,
            conv_layer=conv_layer,
            act_layer=act_layer
        )

        self.init_weights(zcls_pretrained, pretrained_num_classes, num_classes)

    def init_weights(self, pretrained, pretrained_num_classes, num_classes):
        if pretrained != "":
            state_dict = load_state_dict_from_url(pretrained, progress=True)
            self.load_state_dict(state_dict=state_dict, strict=False)
        if num_classes != pretrained_num_classes:
            in_channels = self.head.conv2.in_channels
            self.head.conv2 = nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            self.head.init_weights()

    def train(self, mode: bool = True) -> T:
        super(MobileNetV3Recognizer, self).train(mode=mode)

        if mode and (self.partial_bn or self.fix_bn):
            freezing_bn(self, partial_bn=self.partial_bn)

        return self

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)

        return {KEY_OUTPUT: x}


@registry.RECOGNIZER.register('MobileNetV3')
def build_mobilenet_v3(cfg):
    # recognizer
    zcls_pretrained = cfg.MODEL.RECOGNIZER.PRETRAINED
    pretrained_num_classes = cfg.MODEL.RECOGNIZER.PRETRAINED_NUM_CLASSES
    # backbone
    arch = cfg.MODEL.BACKBONE.ARCH
    in_planes = cfg.MODEL.BACKBONE.IN_PLANES
    # head
    num_classes = cfg.MODEL.HEAD.NUM_CLASSES
    # bn
    fix_bn = cfg.MODEL.NORM.FIX_BN
    partial_bn = cfg.MODEL.NORM.PARTIAL_BN
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
    sigmoid_type = cfg.MODEL.ACT.SIGMOID_TYPE

    return MobileNetV3Recognizer(
        arch=arch,
        in_planes=in_planes,
        width_multiplier=width_multiplier,
        round_nearest=round_nearest,
        reduction=reduction,
        with_attention=with_attention,
        attention_type=attention_type,
        num_classes=num_classes,
        zcls_pretrained=zcls_pretrained,
        pretrained_num_classes=pretrained_num_classes,
        fix_bn=fix_bn,
        partial_bn=partial_bn,
        conv_layer=conv_layer,
        norm_layer=norm_layer,
        act_layer=act_layer,
        sigmoid_type=sigmoid_type
    )

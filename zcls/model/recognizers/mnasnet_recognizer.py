# -*- coding: utf-8 -*-

"""
@date: 2020/12/24 下午7:38
@file: shufflenetv1_recognizer.py
@author: zj
@description: 
"""
from abc import ABC

import torch.nn as nn
from torchvision.models.mnasnet import mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3
from torch.nn.modules.module import T
from torchvision.models.utils import load_state_dict_from_url

from zcls.config.key_word import KEY_OUTPUT
from .. import registry
from ..backbones.mnasnet_backbone import MNASNetBackbone, _round_to_multiple_of
from ..heads.general_head_2d import GeneralHead2D
from ..norm_helper import get_norm, freezing_bn
from ..conv_helper import get_conv
from ..act_helper import get_act

arch_settings = {
    'mnasnet0_5': 0.5,
    'mnasnet0_75': 0.75,
    'mnasnet1_0': 1.0,
    'mnasnet1_3': 1.3
}


class MNASNetRecognizer(nn.Module, ABC):

    def __init__(self,
                 # 输入通道数
                 in_planes=3,
                 # 宽度乘法器
                 width_multiplier=1.,
                 # 设置每一层通道数均为8的倍数
                 round_nearest=8,
                 # 衰减率
                 reduction=4,
                 # 注意力模块类型
                 attention_type='SqueezeAndExcitationBlock2D',
                 # 随机失活概率
                 dropout_rate=0.2,
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
                 act_layer=None
                 ):
        super(MNASNetRecognizer, self).__init__()
        self.fix_bn = fix_bn
        self.partial_bn = partial_bn

        self.backbone = MNASNetBackbone(
            in_planes=in_planes,
            width_multiplier=width_multiplier,
            round_nearest=round_nearest,
            reduction=reduction,
            attention_type=attention_type,
            conv_layer=conv_layer,
            norm_layer=norm_layer,
            act_layer=act_layer
        )
        self.head = GeneralHead2D(
            feature_dims=self.backbone.get_feature_dims(),
            dropout_rate=dropout_rate,
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
        super(MNASNetRecognizer, self).train(mode=mode)

        if mode and (self.partial_bn or self.fix_bn):
            freezing_bn(self, partial_bn=self.partial_bn)

        return self

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)

        return {KEY_OUTPUT: x}


class TorchvisionMNASNet(nn.Module):

    def __init__(self,
                 # 架构
                 arch='mnasnet0_5',
                 # 类别数
                 num_classes=1000,
                 # 预训练模型
                 torchvision_pretrained=False,
                 # 假定预训练模型类别数
                 pretrained_num_classes=1000,
                 # 固定BN
                 fix_bn=False,
                 # 仅训练第一层BN
                 partial_bn=False,
                 ):
        super(TorchvisionMNASNet, self).__init__()

        self.fix_bn = fix_bn
        self.partial_bn = partial_bn

        if arch == 'mnasnet0_5':
            self.model = mnasnet0_5(pretrained=torchvision_pretrained, num_classes=pretrained_num_classes)
        elif arch == 'mnasnet0_75':
            self.model = mnasnet0_75(pretrained=torchvision_pretrained, num_classes=pretrained_num_classes)
        elif arch == 'mnasnet1_0':
            self.model = mnasnet1_0(pretrained=torchvision_pretrained, num_classes=pretrained_num_classes)
        elif arch == 'mnasnet1_3':
            self.model = mnasnet1_3(pretrained=torchvision_pretrained, num_classes=pretrained_num_classes)
        else:
            raise ValueError('no such value')

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
        super(TorchvisionMNASNet, self).train(mode=mode)

        if mode and (self.partial_bn or self.fix_bn):
            freezing_bn(self, partial_bn=self.partial_bn)

        return self

    def forward(self, x):
        x = self.model(x)

        return {KEY_OUTPUT: x}


@registry.RECOGNIZER.register('MNASNet')
def build_mnasnet(cfg):
    # recognizer
    recognizer_name = cfg.MODEL.RECOGNIZER.NAME
    zcls_pretrained = cfg.MODEL.RECOGNIZER.PRETRAINED
    torchvision_pretrained = cfg.MODEL.RECOGNIZER.TORCHVISION_PRETRAINED
    pretrained_num_classes = cfg.MODEL.RECOGNIZER.PRETRAINED_NUM_CLASSES
    # backbone
    arch = cfg.MODEL.BACKBONE.ARCH
    in_planes = cfg.MODEL.BACKBONE.IN_PLANES
    # head
    dropout_rate = cfg.MODEL.HEAD.DROPOUT
    num_classes = cfg.MODEL.HEAD.NUM_CLASSES
    # bn
    fix_bn = cfg.MODEL.NORM.FIX_BN
    partial_bn = cfg.MODEL.NORM.PARTIAL_BN
    norm_layer = get_norm(cfg)
    # compression
    width_multiplier = cfg.MODEL.COMPRESSION.WIDTH_MULTIPLIER
    round_nearest = cfg.MODEL.COMPRESSION.ROUND_NEAREST
    # attention
    reduction = cfg.MODEL.ATTENTION.REDUCTION
    attention_type = cfg.MODEL.ATTENTION.ATTENTION_TYPE
    # conv
    conv_layer = get_conv(cfg)
    # act
    act_layer = get_act(cfg)

    if recognizer_name == 'TorchvisionMNASNet':
        return TorchvisionMNASNet(
            arch=arch,
            num_classes=num_classes,
            torchvision_pretrained=torchvision_pretrained,
            pretrained_num_classes=pretrained_num_classes,
            fix_bn=fix_bn,
            partial_bn=partial_bn
        )
    elif recognizer_name == 'CustomMNASNet':
        return MNASNetRecognizer(
            in_planes=in_planes,
            width_multiplier=width_multiplier,
            round_nearest=round_nearest,
            reduction=reduction,
            attention_type=attention_type,
            dropout_rate=dropout_rate,
            num_classes=num_classes,
            zcls_pretrained=zcls_pretrained,
            pretrained_num_classes=pretrained_num_classes,
            fix_bn=fix_bn,
            partial_bn=partial_bn,
            conv_layer=conv_layer,
            norm_layer=norm_layer,
            act_layer=act_layer
        )
    else:
        raise ValueError(f'{recognizer_name} does not exist')

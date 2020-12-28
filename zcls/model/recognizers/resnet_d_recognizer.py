# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午2:37
@file: resnet_recognizer.py
@author: zj
@description: 
"""
from abc import ABC

import torch.nn as nn
from torch.nn.modules.module import T
from torchvision.models.utils import load_state_dict_from_url

from zcls.config.key_word import KEY_OUTPUT
from .. import registry
from ..backbones.basicblock import BasicBlock
from ..backbones.bottleneck import Bottleneck
from ..backbones.resnet_d_backbone import ResNetDBackbone
from ..heads.resnet_d_head import ResNetDHead
from ..norm_helper import get_norm, freezing_bn
from ..act_helper import get_act
from ..conv_helper import get_conv

arch_settings = {
    'resnet18': (BasicBlock, (2, 2, 2, 2)),
    'resnet34': (BasicBlock, (3, 4, 6, 3)),
    'resnet50': (Bottleneck, (3, 4, 6, 3)),
    'resnet101': (Bottleneck, (3, 4, 23, 3)),
    'resnet152': (Bottleneck, (3, 8, 36, 3))
}


class ResNetDRecognizer(nn.Module, ABC):

    def __init__(self,
                 ##################### for RECOGNIZER
                 arch='resnet18',
                 # zcls预训练模型
                 pretrained="",
                 # 预训练模型类别数
                 pretrained_num_classes=1000,
                 # 固定BN
                 fix_bn=False,
                 # 仅训练第一层BN
                 partial_bn=False,
                 ##################### for HEAD
                 # 随机失活概率
                 dropout_rate=0.,
                 # 输出类别数
                 num_classes=1000,
                 ##################### for BACKBONE
                 # 输入通道数
                 in_planes=3,
                 # 基础通道数,
                 base_planes=64,
                 # 每一层通道数
                 layer_planes=(64, 128, 256, 512),
                 # 是否执行空间下采样
                 down_samples=(0, 1, 1, 1),
                 # cardinality
                 groups=1,
                 # 每组的宽度
                 width_per_group=64,
                 # 卷积层类型
                 conv_layer=None,
                 # 归一化层类型
                 norm_layer=None,
                 # 激活层类型
                 act_layer=None,
                 # 零初始化残差连接
                 zero_init_residual=False
                 ):
        super(ResNetDRecognizer, self).__init__()
        assert arch in arch_settings.keys()
        self.fix_bn = fix_bn
        self.partial_bn = partial_bn

        block_layer, layer_blocks = arch_settings[arch]

        self.backbone = ResNetDBackbone(
            in_planes=in_planes,
            base_planes=base_planes,
            layer_planes=layer_planes,
            layer_blocks=layer_blocks,
            down_samples=down_samples,
            groups=groups,
            width_per_group=width_per_group,
            block_layer=block_layer,
            conv_layer=conv_layer,
            norm_layer=norm_layer,
            act_layer=act_layer,
            zero_init_residual=zero_init_residual
        )
        feature_dims = block_layer.expansion * layer_planes[-1]
        self.head = ResNetDHead(
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

    def train(self, mode: bool = True) -> T:
        super(ResNetDRecognizer, self).train(mode=mode)

        if mode and (self.partial_bn or self.fix_bn):
            freezing_bn(self, partial_bn=self.partial_bn)

        return self

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)

        return {KEY_OUTPUT: x}


@registry.RECOGNIZER.register('ResNetD')
def build_resnet_d(cfg):
    # for recognizer
    pretrained_num_classes = cfg.MODEL.RECOGNIZER.PRETRAINED_NUM_CLASSES
    fix_bn = cfg.MODEL.NORM.FIX_BN
    partial_bn = cfg.MODEL.NORM.PARTIAL_BN
    pretrained = cfg.MODEL.RECOGNIZER.PRETRAINED
    conv_layer = get_conv(cfg)
    norm_layer = get_norm(cfg)
    act_layer = get_act(cfg)
    zero_init_residual = cfg.MODEL.RECOGNIZER.ZERO_INIT_RESIDUAL
    # for backbone
    arch = cfg.MODEL.BACKBONE.ARCH
    in_planes = cfg.MODEL.BACKBONE.IN_PLANES
    base_planes = cfg.MODEL.BACKBONE.BASE_PLANES
    layer_planes = cfg.MODEL.BACKBONE.LAYER_PLANES
    down_samples = cfg.MODEL.BACKBONE.DOWN_SAMPLES
    groups = cfg.MODEL.BACKBONE.GROUPS
    width_per_group = cfg.MODEL.BACKBONE.WIDTH_PER_GROUP
    # for head
    dropout_rate = cfg.MODEL.HEAD.DROPOUT
    num_classes = cfg.MODEL.HEAD.NUM_CLASSES

    return ResNetDRecognizer(
        # for RECOGNIZER
        arch=arch,
        pretrained=pretrained,
        pretrained_num_classes=pretrained_num_classes,
        fix_bn=fix_bn,
        partial_bn=partial_bn,
        # for HEAD
        drdropout_rate=dropout_rate,
        num_classes=num_classes,
        # for BACKBONE
        in_planes=in_planes,
        base_planes=base_planes,
        layer_planes=layer_planes,
        down_samples=down_samples,
        groups=groups,
        width_per_group=width_per_group,
        conv_layer=conv_layer,
        norm_layer=norm_layer,
        act_layer=act_layer,
        zero_init_residual=zero_init_residual
    )

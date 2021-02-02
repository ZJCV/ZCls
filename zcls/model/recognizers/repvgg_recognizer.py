# -*- coding: utf-8 -*-

"""
@date: 2021/2/2 下午5:19
@file: repvgg_recognizer.py
@author: zj
@description: RegVGG，参考[RepVGG/repvgg.py](https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py)
"""

import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url

from zcls.config.key_word import KEY_OUTPUT
from .. import registry
from ..backbones.repvgg_backbone import RepVGGBackbone
from ..heads.general_head_2d import GeneralHead2D
from ..act_helper import get_act
from ..conv_helper import get_conv

optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}

arch_settings = {
    # name: (num_blocks, width_multiplier, groups)
    'repvgg_a0': ((2, 4, 14, 1), (0.75, 2.5), {}),
    'repvgg_a1': ((2, 4, 14, 1), (1, 2.5), {}),
    'repvgg_a2': ((2, 4, 14, 1), (1.5, 2.75), {}),
    'repvgg_b0': ((4, 6, 16, 1), (1, 2.5), {}),
    'repvgg_b1': ((4, 6, 16, 1), (2, 4), {}),
    'repvgg_b1g2': ((4, 6, 16, 1), (2, 4), g2_map),
    'repvgg_b1g4': ((4, 6, 16, 1), (2, 4), g4_map),
    'repvgg_b2': ((4, 6, 16, 1), (2.5, 5), {}),
    'repvgg_b2g2': ((4, 6, 16, 1), (2.5, 5), g2_map),
    'repvgg_b2g4': ((4, 6, 16, 1), (2.5, 5), g4_map),
    'repvgg_b3': ((4, 6, 16, 1), (3, 5), {}),
    'repvgg_b3g2': ((4, 6, 16, 1), (3, 5), g2_map),
    'repvgg_b3g4': ((4, 6, 16, 1), (3, 5), g4_map),
}


class RepVGGRecognizer(nn.Module):

    def __init__(self,
                 # 架构
                 arch='repvgg_b1g4',
                 ############## RECOGNIZER
                 # zcls预训练模型
                 pretrained="",
                 # 预训练模型类别数
                 pretrained_num_classes=1000,
                 ############## backbone
                 # 输入通道数
                 in_channels=3,
                 # 基础通道数,
                 base_channels=64,
                 # 每一层通道数
                 layer_planes=(64, 128, 256, 512),
                 # 是否执行空间下采样
                 down_samples=(1, 1, 1, 1),
                 # 卷积层类型
                 conv_layer=None,
                 # 激活层类型
                 act_layer=None,
                 ############### head
                 # 类别数
                 num_classes=1000,
                 # 随机失活概率
                 dropout_rate=0.
                 ):
        super(RepVGGRecognizer, self).__init__()
        assert arch in arch_settings.keys()

        num_blocks, width_multipliers, groups = arch_settings[arch]
        self.backbone = RepVGGBackbone(
            in_channels=in_channels,
            base_channels=base_channels,
            layer_planes=layer_planes,
            layer_blocks=num_blocks,
            down_samples=down_samples,
            width_multipliers=width_multipliers,
            groups=groups,
            conv_layer=conv_layer,
            act_layer=act_layer,
        )
        feature_dims = int(width_multipliers[1] * layer_planes[-1])
        self.head = GeneralHead2D(
            feature_dims=feature_dims,
            num_classes=pretrained_num_classes,
            dropout_rate=dropout_rate,
        )

        self.init_weights(pretrained,
                          pretrained_num_classes,
                          num_classes)

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
            self.head.init_weights()

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)

        return {KEY_OUTPUT: x}


@registry.RECOGNIZER.register('RepVGG')
def build_resnet(cfg):
    # for recognizer
    pretrained = cfg.MODEL.RECOGNIZER.PRETRAINED
    pretrained_num_classes = cfg.MODEL.RECOGNIZER.PRETRAINED_NUM_CLASSES
    # for backbone
    arch = cfg.MODEL.BACKBONE.ARCH
    in_planes = cfg.MODEL.BACKBONE.IN_PLANES
    base_planes = cfg.MODEL.BACKBONE.BASE_PLANES
    layer_planes = cfg.MODEL.BACKBONE.LAYER_PLANES
    down_samples = cfg.MODEL.BACKBONE.DOWN_SAMPLES
    conv_layer = get_conv(cfg)
    act_layer = get_act(cfg)
    # for head
    dropout_rate = cfg.MODEL.HEAD.DROPOUT
    num_classes = cfg.MODEL.HEAD.NUM_CLASSES

    return RepVGGRecognizer(
        arch=arch,
        pretrained=pretrained,
        pretrained_num_classes=pretrained_num_classes,
        in_channels=in_planes,
        base_channels=base_planes,
        layer_planes=layer_planes,
        down_samples=down_samples,
        conv_layer=conv_layer,
        act_layer=act_layer,
        dropout_rate=dropout_rate,
        num_classes=num_classes
    )

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
from torchvision.models.resnet import model_urls, resnet18, resnet50, resnext50_32x4d
from torchvision.models.utils import load_state_dict_from_url

from zcls.config.key_word import KEY_OUTPUT
from .. import registry
from ..backbones.basicblock import BasicBlock
from ..backbones.bottleneck import Bottleneck
from ..backbones.resnet_backbone import ResNetBackbone
from ..heads.resnet_head import ResNetHead
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


class ResNetRecognizer(nn.Module, ABC):

    def __init__(self,
                 ##################### for RECOGNIZER
                 arch='resnet18',
                 # zcls预训练模型
                 pretrained="",
                 # torchvision预训练模型
                 torchvision_pretrained=False,
                 # 预训练模型类别数
                 pretrained_num_classes=1000,
                 # 固定BN
                 fix_bn=False,
                 # 仅训练第一层BN
                 partial_bn=False,
                 ##################### for HEAD
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
        super(ResNetRecognizer, self).__init__()
        assert arch in arch_settings.keys()
        self.fix_bn = fix_bn
        self.partial_bn = partial_bn

        block_layer, layer_blocks = arch_settings[arch]

        self.backbone = ResNetBackbone(
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
        self.head = ResNetHead(
            feature_dims=feature_dims,
            num_classes=pretrained_num_classes
        )

        self._init_weights(arch=arch,
                           pretrained=pretrained,
                           torchvision_pretrained=torchvision_pretrained,
                           pretrained_num_classes=pretrained_num_classes,
                           num_classes=num_classes)

    def _init_weights(self,
                      arch,
                      pretrained,
                      torchvision_pretrained,
                      pretrained_num_classes,
                      num_classes
                      ):
        if torchvision_pretrained:
            state_dict = load_state_dict_from_url(model_urls[arch], progress=True)
            self.backbone.load_state_dict(state_dict, strict=False)
            self.head.load_state_dict(state_dict, strict=False)
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
        super(ResNetRecognizer, self).train(mode=mode)

        if mode and (self.partial_bn or self.fix_bn):
            freezing_bn(self, partial_bn=self.partial_bn)

        return self

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)

        return {KEY_OUTPUT: x}


class TorchvisionResNet(nn.Module, ABC):

    def __init__(self,
                 arch="resnet18",
                 num_classes=1000,
                 torchvision_pretrained=False,
                 pretrained_num_classes=1000,
                 fix_bn=False,
                 partial_bn=False):
        super(TorchvisionResNet, self).__init__()

        self.num_classes = num_classes
        self.fix_bn = fix_bn
        self.partial_bn = partial_bn

        if arch == 'resnet18':
            self.model = resnet18(pretrained=torchvision_pretrained, num_classes=pretrained_num_classes)
        elif arch == 'resnet50':
            self.model = resnet50(pretrained=torchvision_pretrained, num_classes=pretrained_num_classes)
        elif arch == 'resnext50_32x4d':
            self.model = resnext50_32x4d(pretrained=torchvision_pretrained, num_classes=pretrained_num_classes)
        else:
            raise ValueError('no such value')

        self._init_weights(num_classes, pretrained_num_classes)

    def _init_weights(self, num_classes, pretrained_num_classes):
        if num_classes != pretrained_num_classes:
            fc = self.model.fc
            fc_features = fc.in_features
            self.model.fc = nn.Linear(fc_features, num_classes)

            nn.init.normal_(self.model.fc.weight, 0, 0.01)
            nn.init.zeros_(self.model.fc.bias)

    def train(self, mode: bool = True) -> T:
        super(TorchvisionResNet, self).train(mode=mode)

        if mode and (self.partial_bn or self.fix_bn):
            freezing_bn(self, partial_bn=self.partial_bn)

        return self

    def forward(self, x):
        x = self.model(x)

        return {KEY_OUTPUT: x}


@registry.RECOGNIZER.register('ResNet')
def build_resnet(cfg):
    # for recognizer
    recognizer_name = cfg.MODEL.RECOGNIZER.NAME
    torchvision_pretrained = cfg.MODEL.TORCHVISION_PRETRAINED
    pretrained_num_classes = cfg.MODEL.RECOGNIZER.PRETRAINED_NUM_CLASSES
    fix_bn = cfg.MODEL.NORM.FIX_BN
    partial_bn = cfg.MODEL.NORM.PARTIAL_BN
    # for backbone
    arch = cfg.MODEL.BACKBONE.ARCH
    # for head
    num_classes = cfg.MODEL.HEAD.NUM_CLASSES

    if recognizer_name == 'TorchvisionResNet':
        return TorchvisionResNet(
            arch=arch,
            num_classes=num_classes,
            torchvision_pretrained=torchvision_pretrained,
            pretrained_num_classes=pretrained_num_classes,
            fix_bn=fix_bn,
            partial_bn=partial_bn
        )
    elif recognizer_name == 'CustomResNet':
        # for recognizer
        pretrained = cfg.MODEL.PRETRAINED
        conv_layer = get_conv(cfg)
        norm_layer = get_norm(cfg)
        act_layer = get_act(cfg)
        zero_init_residual = cfg.MODEL.RECOGNIZER.ZERO_INIT_RESIDUAL
        # for backbone
        in_planes = cfg.MODEL.BACKBONE.IN_PLANES
        base_planes = cfg.MODEL.BACKBONE.BASE_PLANES
        layer_planes = cfg.MODEL.BACKBONE.LAYER_PLANES
        down_samples = cfg.MODEL.BACKBONE.DOWN_SAMPLES
        groups = cfg.MODEL.BACKBONE.GROUPS
        width_per_group = cfg.MODEL.BACKBONE.WITH_PER_GROUP

        return ResNetRecognizer(
            # for RECOGNIZER
            arch=arch,
            pretrained=pretrained,
            torchvision_pretrained=torchvision_pretrained,
            pretrained_num_classes=pretrained_num_classes,
            fix_bn=fix_bn,
            partial_bn=partial_bn,
            # for HEAD
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
    else:
        raise ValueError(f'{recognizer_name} does not exist')

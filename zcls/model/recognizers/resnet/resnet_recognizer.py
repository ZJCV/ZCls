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
from torchvision.models.resnet import resnet18, resnet50, resnext50_32x4d
from torchvision.models.utils import load_state_dict_from_url

from zcls.config.key_word import KEY_OUTPUT
from zcls.model import registry
from zcls.model.backbones.resnet.basicblock import BasicBlock
from zcls.model.backbones.resnet.bottleneck import Bottleneck
from zcls.model.backbones.resnet.sknet_block import SKNetBlock
from zcls.model.backbones.resnet.resnest_block import ResNeStBlock
from zcls.model.backbones.resnet.resnet_backbone import ResNetBackbone
from zcls.model.backbones.resnet.resnet_d_backbone import ResNetDBackbone
from zcls.model.heads.resnet_head import ResNetHead
from zcls.model.heads.resnet_d_head import ResNetDHead
from zcls.model.norm_helper import get_norm, freezing_bn
from zcls.model.act_helper import get_act
from zcls.model.conv_helper import get_conv

arch_settings = {
    # name: (Backbone, Head, Block, Layer_planes, groups, width_per_group)
    'resnet18': (ResNetBackbone, ResNetHead, BasicBlock, (2, 2, 2, 2), 1, 64),
    'resnet34': (ResNetBackbone, ResNetHead, BasicBlock, (3, 4, 6, 3), 1, 64),
    'resnet50': (ResNetBackbone, ResNetHead, Bottleneck, (3, 4, 6, 3), 1, 64),
    'resnet101': (ResNetBackbone, ResNetHead, Bottleneck, (3, 4, 23, 3), 1, 64),
    'resnet152': (ResNetBackbone, ResNetHead, Bottleneck, (3, 8, 36, 3), 1, 64),
    'resnext50_32x4d': (ResNetBackbone, ResNetHead, Bottleneck, (3, 4, 6, 3), 32, 4),
    'resnext101_32x8d': (ResNetBackbone, ResNetHead, Bottleneck, (3, 4, 23, 3), 32, 8),
    # name: (Backbone, Head, Block, Layer_planes, groups, width_per_group)
    'resnetd18': (ResNetDBackbone, ResNetDHead, BasicBlock, (2, 2, 2, 2), 1, 64),
    'resnetd34': (ResNetDBackbone, ResNetDHead, BasicBlock, (3, 4, 6, 3), 1, 64),
    'resnetd50': (ResNetDBackbone, ResNetDHead, Bottleneck, (3, 4, 6, 3), 1, 64),
    'resnetd101': (ResNetDBackbone, ResNetDHead, Bottleneck, (3, 4, 23, 3), 1, 64),
    'resnetd152': (ResNetDBackbone, ResNetDHead, Bottleneck, (3, 8, 36, 3), 1, 64),
    'resnextd50_32x4d': (ResNetDBackbone, ResNetDHead, Bottleneck, (3, 4, 6, 3), 32, 4),
    'resnedxdt101_32x8d': (ResNetDBackbone, ResNetDHead, Bottleneck, (3, 4, 23, 3), 32, 8),
    # name: (Backbone, Head, Block, Layer_planes, groups, width_per_group)
    'sknet50': (ResNetDBackbone, ResNetDHead, SKNetBlock, (3, 4, 6, 3), 1, 64),
    # name: (Backbone, Head, Block, Layer_planes, radix, groups, width_per_group)
    'resnest50_1s1x64d': (ResNetDBackbone, ResNetDHead, ResNeStBlock, (3, 4, 6, 3), 1, 1, 64),
    'resnest50_2s1x64d': (ResNetDBackbone, ResNetDHead, ResNeStBlock, (3, 4, 6, 3), 2, 1, 64),
    'resnest50_4s1x64d': (ResNetDBackbone, ResNetDHead, ResNeStBlock, (3, 4, 6, 3), 4, 1, 64),
    'resnest50_2s2x40d': (ResNetDBackbone, ResNetDHead, ResNeStBlock, (3, 4, 6, 3), 2, 2, 40),
    'resnest50_2s2x40d_fast': (ResNetDBackbone, ResNetDHead, ResNeStBlock, (3, 4, 6, 3), 2, 2, 40)
}


class ResNetRecognizer(nn.Module, ABC):

    def __init__(self,
                 ##################### for RECOGNIZER
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
                 # 架构
                 arch='resnet18',
                 # 输入通道数
                 in_planes=3,
                 # 基础通道数,
                 base_planes=64,
                 # 每一层通道数
                 layer_planes=(64, 128, 256, 512),
                 # 是否执行空间下采样
                 down_samples=(0, 1, 1, 1),
                 # 是否使用注意力模块
                 with_attentions=(0, 0, 0, 0),
                 # 衰减率
                 reduction=16,
                 # 注意力模块类型
                 attention_type='GlobalContextBlock2D',
                 # 卷积层类型
                 conv_layer=None,
                 # 归一化层类型
                 norm_layer=None,
                 # 激活层类型
                 act_layer=None,
                 # 零初始化残差连接
                 zero_init_residual=False,
                 # 是否使用AvgPool进行下采样
                 use_avg=False,
                 # 在3x3之前执行下采样操作
                 fast_avg=False
                 ):
        super(ResNetRecognizer, self).__init__()
        assert arch in arch_settings.keys()
        self.fix_bn = fix_bn
        self.partial_bn = partial_bn

        radix = 1
        if 'resnest' in arch:
            backbone_layer, head_layer, block_layer, layer_blocks, radix, groups, width_per_group = arch_settings[arch]
            if 'fast' in arch:
                fast_avg = True
        else:
            backbone_layer, head_layer, block_layer, layer_blocks, groups, width_per_group = arch_settings[arch]

        self.backbone = backbone_layer(
            in_planes=in_planes,
            base_planes=base_planes,
            layer_planes=layer_planes,
            layer_blocks=layer_blocks,
            down_samples=down_samples,
            groups=groups,
            width_per_group=width_per_group,
            with_attentions=with_attentions,
            reduction=reduction,
            attention_type=attention_type,
            block_layer=block_layer,
            conv_layer=conv_layer,
            norm_layer=norm_layer,
            act_layer=act_layer,
            zero_init_residual=zero_init_residual,
            radix=radix,
            use_avg=use_avg,
            fast_avg=fast_avg
        )
        feature_dims = block_layer.expansion * layer_planes[-1]
        self.head = head_layer(
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
                 partial_bn=False,
                 zero_init_residual=False):
        super(TorchvisionResNet, self).__init__()

        self.num_classes = num_classes
        self.fix_bn = fix_bn
        self.partial_bn = partial_bn

        if arch == 'resnet18':
            self.model = resnet18(pretrained=torchvision_pretrained, num_classes=pretrained_num_classes,
                                  zero_init_residual=zero_init_residual)
        elif arch == 'resnet50':
            self.model = resnet50(pretrained=torchvision_pretrained, num_classes=pretrained_num_classes,
                                  zero_init_residual=zero_init_residual)
        elif arch == 'resnext50_32x4d':
            self.model = resnext50_32x4d(pretrained=torchvision_pretrained, num_classes=pretrained_num_classes,
                                         zero_init_residual=zero_init_residual)
        else:
            raise ValueError('no such value')

        self.init_weights(num_classes, pretrained_num_classes)

    def init_weights(self, num_classes, pretrained_num_classes):
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
    torchvision_pretrained = cfg.MODEL.RECOGNIZER.TORCHVISION_PRETRAINED
    pretrained = cfg.MODEL.RECOGNIZER.PRETRAINED
    pretrained_num_classes = cfg.MODEL.RECOGNIZER.PRETRAINED_NUM_CLASSES
    fix_bn = cfg.MODEL.NORM.FIX_BN
    partial_bn = cfg.MODEL.NORM.PARTIAL_BN
    # for backbone
    arch = cfg.MODEL.BACKBONE.ARCH
    in_planes = cfg.MODEL.BACKBONE.IN_PLANES
    base_planes = cfg.MODEL.BACKBONE.BASE_PLANES
    layer_planes = cfg.MODEL.BACKBONE.LAYER_PLANES
    down_samples = cfg.MODEL.BACKBONE.DOWN_SAMPLES
    conv_layer = get_conv(cfg)
    norm_layer = get_norm(cfg)
    act_layer = get_act(cfg)
    zero_init_residual = cfg.MODEL.RECOGNIZER.ZERO_INIT_RESIDUAL
    use_avg = cfg.MODEL.BACKBONE.USE_AVG
    fast_avg = cfg.MODEL.BACKBONE.FAST_AVG
    # for head
    dropout_rate = cfg.MODEL.HEAD.DROPOUT
    num_classes = cfg.MODEL.HEAD.NUM_CLASSES
    # for attention
    with_attentions = cfg.MODEL.ATTENTION.WITH_ATTENTIONS
    reduction = cfg.MODEL.ATTENTION.REDUCTION
    attention_type = cfg.MODEL.ATTENTION.ATTENTION_TYPE

    if recognizer_name == 'TorchvisionResNet':
        return TorchvisionResNet(
            arch=arch,
            num_classes=num_classes,
            torchvision_pretrained=torchvision_pretrained,
            pretrained_num_classes=pretrained_num_classes,
            fix_bn=fix_bn,
            partial_bn=partial_bn,
            zero_init_residual=zero_init_residual
        )
    elif recognizer_name == 'ZClsResNet':
        return ResNetRecognizer(
            # for RECOGNIZER
            pretrained=pretrained,
            pretrained_num_classes=pretrained_num_classes,
            fix_bn=fix_bn,
            partial_bn=partial_bn,
            # for HEAD
            dropout_rate=dropout_rate,
            num_classes=num_classes,
            # for BACKBONE
            arch=arch,
            in_planes=in_planes,
            base_planes=base_planes,
            layer_planes=layer_planes,
            down_samples=down_samples,
            with_attentions=with_attentions,
            reduction=reduction,
            attention_type=attention_type,
            conv_layer=conv_layer,
            norm_layer=norm_layer,
            act_layer=act_layer,
            zero_init_residual=zero_init_residual,
            use_avg=use_avg,
            fast_avg=fast_avg
        )
    else:
        raise ValueError(f'{recognizer_name} does not exist')

# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午2:37
@file: resnet_recognizer.py
@author: zj
@description: 
"""

import torch.nn as nn
from torch.nn.modules.module import T
from torchvision.models.utils import load_state_dict_from_url

from .. import registry
from ..backbones.basicblock import BasicBlock
from ..backbones.bottleneck import Bottleneck
from ..backbones.resnet_d_backbone import ResNetDBackbone
from ..heads.resnet_head import ResNetHead
from ..norm_helper import get_norm, freezing_bn

arch_settings = {
    'resnet18': (BasicBlock, (2, 2, 2, 2)),
    'resnet34': (BasicBlock, (3, 4, 6, 3)),
    'resnet50': (Bottleneck, (3, 4, 6, 3)),
    'resnet101': (Bottleneck, (3, 4, 23, 3)),
    'resnet152': (Bottleneck, (3, 8, 36, 3))
}


class ResNetDRecognizer(nn.Module):

    def __init__(self,
                 arch='resnet18',
                 feature_dims=2048,
                 num_classes=1000,
                 groups=1,
                 width_per_group=64,
                 zcls_pretrained="",
                 pretrained_num_classes=1000,
                 fix_bn=False,
                 partial_bn=False,
                 norm_layer=None):
        super(ResNetDRecognizer, self).__init__()
        assert isinstance(zcls_pretrained, str)

        self.num_classes = num_classes
        self.fix_bn = fix_bn
        self.partial_bn = partial_bn

        block_layer, layer_blocks = arch_settings[arch]

        self.backbone = ResNetDBackbone(
            layer_blocks=layer_blocks,
            groups=groups,
            width_per_group=width_per_group,
            block_layer=block_layer,
            norm_layer=norm_layer
        )
        self.head = ResNetHead(
            feature_dims=feature_dims,
            num_classes=pretrained_num_classes
        )

        self._init_weights(pretrained=zcls_pretrained, pretrained_num_classes=pretrained_num_classes)

    def _init_weights(self, pretrained="", pretrained_num_classes=1000):
        if pretrained != "":
            state_dict = load_state_dict_from_url(pretrained, progress=True)
            self.backbone.load_state_dict(state_dict, strict=False)
            self.head.load_state_dict(state_dict, strict=False)
        if self.num_classes != pretrained_num_classes:
            fc = self.head.fc
            fc_features = fc.in_features
            self.head.fc = nn.Linear(fc_features, self.num_classes)

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

        return {'probs': x}


@registry.RECOGNIZER.register('ResNetD')
def build_resnet_d(cfg):
    type = cfg.MODEL.RECOGNIZER.NAME
    zcls_pretrained = cfg.MODEL.ZClS_PRETRAINED
    pretrained_num_classes = cfg.MODEL.PRETRAINED_NUM_CLASSES
    arch = cfg.MODEL.BACKBONE.ARCH
    num_classes = cfg.MODEL.HEAD.NUM_CLASSES
    fix_bn = cfg.MODEL.NORM.FIX_BN
    partial_bn = cfg.MODEL.NORM.PARTIAL_BN

    feature_dims = cfg.MODEL.HEAD.FEATURE_DIMS
    norm_layer = get_norm(cfg)

    return ResNetDRecognizer(
        arch=arch,
        feature_dims=feature_dims,
        num_classes=num_classes,
        zcls_pretrained=zcls_pretrained,
        pretrained_num_classes=pretrained_num_classes,
        fix_bn=fix_bn,
        partial_bn=partial_bn,
        norm_layer=norm_layer
    )

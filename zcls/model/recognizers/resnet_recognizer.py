# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午2:37
@file: resnet_recognizer.py
@author: zj
@description: 
"""

import torch.nn as nn
from torch.nn.modules.module import T
from torchvision.models import resnet
from torchvision.models.utils import load_state_dict_from_url

from .. import registry
from ..backbones.basicblock import BasicBlock
from ..backbones.bottleneck import Bottleneck
from ..backbones.resnet_backbone import ResNetBackbone
from ..heads.resnet_head import ResNetHead
from ..norm_helper import get_norm, freezing_bn

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

arch_settings = {
    'resnet18': (BasicBlock, (2, 2, 2, 2)),
    'resnet34': (BasicBlock, (3, 4, 6, 3)),
    'resnet50': (Bottleneck, (3, 4, 6, 3)),
    'resnet101': (Bottleneck, (3, 4, 23, 3)),
    'resnet152': (Bottleneck, (3, 8, 36, 3))
}


class ResNetRecognizer(nn.Module):

    def __init__(self,
                 arch='resnet18',
                 feature_dims=2048,
                 num_classes=1000,
                 groups=1,
                 width_per_group=64,
                 torchvision_pretrained=False,
                 fix_bn=False,
                 partial_bn=False,
                 norm_layer=None):
        super(ResNetRecognizer, self).__init__()

        self.num_classes = num_classes
        self.fix_bn = fix_bn
        self.partial_bn = partial_bn

        block_layer, layer_blocks = arch_settings[arch]

        self.backbone = ResNetBackbone(
            layer_blocks=layer_blocks,
            groups=groups,
            width_per_group=width_per_group,
            block_layer=block_layer,
            norm_layer=norm_layer
        )
        self.head = ResNetHead(
            feature_dims=feature_dims,
            num_classes=1000
        )

        self._init_weights(arch=arch, pretrained=torchvision_pretrained)

    def _init_weights(self, arch='resnet18', pretrained=False):
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls[arch], progress=True)
            self.backbone.load_state_dict(state_dict, strict=False)
            self.head.load_state_dict(state_dict, strict=False)
        if self.num_classes != 1000:
            fc = self.head.fc
            fc_features = fc.in_features
            self.head.fc = nn.Linear(fc_features, self.num_classes)

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

        return {'probs': x}


class ResNet_Pytorch(nn.Module):

    def __init__(self,
                 arch="resnet18",
                 num_classes=1000,
                 torchvision_pretrained=False,
                 fix_bn=False,
                 partial_bn=False):
        super(ResNet_Pytorch, self).__init__()

        self.num_classes = num_classes
        self.fix_bn = fix_bn
        self.partial_bn = partial_bn

        if arch == 'resnet18':
            self.model = resnet.resnet18(pretrained=torchvision_pretrained, num_classes=1000)
        elif arch == 'resnet50':
            self.model = resnet.resnet50(pretrained=torchvision_pretrained, num_classes=1000)
        elif arch == 'resnext50_32x4d':
            self.model = resnet.resnext50_32x4d(pretrained=torchvision_pretrained, num_classes=1000)
        else:
            raise ValueError('no such value')

        self._init_weights()

    def _init_weights(self):
        if self.num_classes != 1000:
            fc = self.model.fc
            fc_features = fc.in_features
            self.model.fc = nn.Linear(fc_features, self.num_classes)

            nn.init.normal_(self.model.fc.weight, 0, 0.01)
            nn.init.zeros_(self.model.fc.bias)

    def train(self, mode: bool = True) -> T:
        super(ResNet_Pytorch, self).train(mode=mode)

        if mode and (self.partial_bn or self.fix_bn):
            freezing_bn(self, partial_bn=self.partial_bn)

        return self

    def forward(self, x):
        x = self.model(x)

        return {'probs': x}


@registry.RECOGNIZER.register('ResNet')
def build_resnet(cfg):
    type = cfg.MODEL.RECOGNIZER.NAME
    torchvision_pretrained = cfg.MODEL.TORCHVISION_PRETRAINED
    arch = cfg.MODEL.BACKBONE.ARCH
    num_classes = cfg.MODEL.HEAD.NUM_CLASSES
    fix_bn = cfg.MODEL.NORM.FIX_BN
    partial_bn = cfg.MODEL.NORM.PARTIAL_BN

    if type == 'ResNet_Pytorch':
        return ResNet_Pytorch(
            arch=arch,
            num_classes=num_classes,
            torchvision_pretrained=torchvision_pretrained,
            fix_bn=fix_bn,
            partial_bn=partial_bn
        )
    elif type == 'ResNet_Custom':
        feature_dims = cfg.MODEL.HEAD.FEATURE_DIMS
        norm_layer = get_norm(cfg)

        return ResNetRecognizer(
            arch=arch,
            feature_dims=feature_dims,
            num_classes=num_classes,
            torchvision_pretrained=torchvision_pretrained,
            fix_bn=fix_bn,
            partial_bn=partial_bn,
            norm_layer=norm_layer
        )
    else:
        raise ValueError(f'{type} does not exist')

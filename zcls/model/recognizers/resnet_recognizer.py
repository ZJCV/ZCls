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
                 torchvision_pretrained=False,
                 fix_bn=False,
                 partial_bn=False):
        super(ResNetRecognizer, self).__init__()
        block_layer, layer_blocks = arch_settings[arch]

        self.backbone = ResNetBackbone(
            layer_blocks=layer_blocks,
            block_layer=block_layer
        )
        self.head = ResNetHead(
            feature_dims=feature_dims,
            num_classes=num_classes
        )

        self._init_weights(arch=arch, pretrained=torchvision_pretrained)
        self.fix_bn = fix_bn
        self.partial_bn = partial_bn

    def _init_weights(self, arch='resnet18', pretrained=False):
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls[arch], progress=True)
            res = self.backbone.load_state_dict(state_dict, strict=False)
            print(res)
            res = self.head.load_state_dict(state_dict, strict=False)
            print(res)

    def freezing_bn(self: T) -> None:
        count = 0
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                count += 1
                if count == 1 and self.partial_bn:
                    continue

                m.eval()
                # shutdown update in frozen mode
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def train(self, mode: bool = True) -> T:
        super(ResNetRecognizer, self).train(mode=mode)

        if mode and (self.partial_bn or self.fix_bn):
            self.freezing_bn()

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
        if arch == 'resnet18':
            self.model = resnet.resnet18(pretrained=torchvision_pretrained, num_classes=num_classes)
        elif arch == 'resnet50':
            self.model = resnet.resnet50(pretrained=torchvision_pretrained, num_classes=num_classes)
        else:
            raise ValueError('no such value')
        self.fix_bn = fix_bn
        self.partial_bn = partial_bn

    def freezing_bn(self):
        count = 0
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                count += 1
                if count == 1 and self.partial_bn:
                    continue

                m.eval()
                # shutdown update in frozen mode
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def train(self, mode: bool = True) -> T:
        super(ResNet_Pytorch, self).train(mode=mode)

        if mode and (self.partial_bn or self.fix_bn):
            self.freezing_bn()

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

    if type == 'ResNet_Pytorch':
        return ResNet_Pytorch(
            arch=arch,
            num_classes=num_classes,
            torchvision_pretrained=torchvision_pretrained
        )
    elif type == 'ResNet_Custom':
        feature_dims = cfg.MODEL.HEAD.FEATURE_DIMS
        return ResNetRecognizer(
            arch=arch,
            feature_dims=feature_dims,
            num_classes=num_classes,
            torchvision_pretrained=torchvision_pretrained
        )
    else:
        raise ValueError(f'{type} does not exist')

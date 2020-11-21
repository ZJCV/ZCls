# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午2:37
@file: resnet_recognizer.py
@author: zj
@description: 
"""

import torch.nn as nn
from torchvision.models import resnet

from .. import registry
from ..backbones.basicblock import BasicBlock
from ..backbones.bottleneck import Bottleneck
from ..backbones.resnet_backbone import ResNetBackbone
from ..heads.resnet_head import ResNetHead

arch_settings = {
    18: (BasicBlock, (2, 2, 2, 2)),
    34: (BasicBlock, (3, 4, 6, 3)),
    50: (Bottleneck, (3, 4, 6, 3)),
    101: (Bottleneck, (3, 4, 23, 3)),
    152: (Bottleneck, (3, 8, 36, 3))
}


class ResNetRecognizer(nn.Module):

    def __init__(self,
                 arch=50,
                 feature_dims=2048,
                 num_classes=1000):
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

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)

        return x

@registry.RECOGNIZER.register('ResNet')
def build_resnet(cfg):
    type = cfg.MODEL.RECOGNIZER.NAME

    if type == 'R50_Pytorch':
        return resnet.resnet50()
    elif type == 'ResNet_Custom':
        arch = cfg.MODEL.BACKBONE.ARCH
        feature_dims = cfg.MODEL.HEAD.FEATURE_DIMS
        num_classes = cfg.MODEL.HEAD.NUM_CLASSES
        return ResNetRecognizer(
            arch=arch,
            feature_dims=feature_dims,
            num_classes=num_classes
        )
    else:
        raise ValueError(f'{type} does not exist')

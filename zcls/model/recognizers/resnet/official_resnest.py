# -*- coding: utf-8 -*-

"""
@date: 2021/1/7 下午7:51
@file: official_resnest.py
@author: zj
@description: 
"""
from abc import ABC

import torch
import torch.nn as nn
from torch.nn.modules.module import T
from resnest.torch.resnest import resnest50, resnest101, resnest200, resnest269
from resnest.torch.resnet import ResNet, Bottleneck

from zcls.config.key_word import KEY_OUTPUT
from zcls.model import registry
from zcls.model.norm_helper import freezing_bn


class OfficialResNeSt(nn.Module, ABC):

    def __init__(self,
                 arch='resnest50_2s2x40d',
                 dropout_rate=0.,
                 num_classes=1000,
                 fix_bn=False,
                 partial_bn=False,
                 pretrained="",
                 pretrained_num_classes=1000,
                 ):
        super(OfficialResNeSt, self).__init__()

        self.num_classes = num_classes
        self.fix_bn = fix_bn
        self.partial_bn = partial_bn

        if arch == 'resnest50':
            self.model = resnest50(num_classes=pretrained_num_classes,
                                   final_drop=dropout_rate
                                   )
        elif arch == 'resnest50_2s2x40d':
            radix = 2
            groups = 2
            width_per_group = 40
            avg_first = False
            self.model = ResNet(Bottleneck,
                                [3, 4, 6, 3],
                                radix=radix,
                                groups=groups,
                                bottleneck_width=width_per_group,
                                deep_stem=True,
                                stem_width=32,
                                avg_down=True,
                                avd=True,
                                avd_first=avg_first,
                                final_drop=dropout_rate,
                                num_classes=pretrained_num_classes
                                )
        elif arch == 'resnest50_fast_2s2x40d':
            radix = 2
            groups = 2
            width_per_group = 40
            avg_first = True
            self.model = ResNet(Bottleneck,
                                [3, 4, 6, 3],
                                radix=radix,
                                groups=groups,
                                bottleneck_width=width_per_group,
                                deep_stem=True,
                                stem_width=32,
                                avg_down=True,
                                avd=True,
                                avd_first=avg_first,
                                final_drop=dropout_rate,
                                num_classes=pretrained_num_classes
                                )
        elif arch == 'resnest50_fast_2s1x64d':
            radix = 2
            groups = 1
            width_per_group = 64
            avg_first = True
            self.model = ResNet(Bottleneck,
                                [3, 4, 6, 3],
                                radix=radix,
                                groups=groups,
                                bottleneck_width=width_per_group,
                                deep_stem=True,
                                stem_width=32,
                                avg_down=True,
                                avd=True,
                                avd_first=avg_first,
                                final_drop=dropout_rate,
                                num_classes=pretrained_num_classes
                                )
        elif arch == 'resnest101':
            self.model = resnest101(num_classes=pretrained_num_classes,
                                    final_drop=dropout_rate
                                    )
        elif arch == 'resnest200':
            self.model = resnest200(num_classes=pretrained_num_classes,
                                    final_drop=dropout_rate
                                    )
        elif arch == 'resnest269':
            self.model = resnest269(num_classes=pretrained_num_classes,
                                    final_drop=dropout_rate
                                    )
        else:
            raise ValueError('no such value')

        self.init_weights(pretrained,
                          pretrained_num_classes,
                          num_classes)

    def init_weights(self, pretrained, pretrained_num_classes, num_classes):
        if pretrained != "":
            self.model.load_state_dict(torch.load(pretrained))
        if num_classes != pretrained_num_classes:
            fc = self.model.fc
            fc_features = fc.in_features

            fc = nn.Linear(fc_features, num_classes)
            nn.init.normal_(fc.weight, 0, 0.01)
            nn.init.zeros_(fc.bias)

            self.model.fc = fc

    def train(self, mode: bool = True) -> T:
        super(OfficialResNeSt, self).train(mode=mode)

        if mode and (self.partial_bn or self.fix_bn):
            freezing_bn(self, partial_bn=self.partial_bn)

        return self

    def forward(self, x):
        x = self.model(x)

        return {KEY_OUTPUT: x}


@registry.RECOGNIZER.register('OfficialResNeSt')
def build_official_resnest(cfg):
    # for recognizer
    pretrained = cfg.MODEL.RECOGNIZER.PRETRAINED
    pretrained_num_classes = cfg.MODEL.RECOGNIZER.PRETRAINED_NUM_CLASSES
    fix_bn = cfg.MODEL.NORM.FIX_BN
    partial_bn = cfg.MODEL.NORM.PARTIAL_BN
    # for backbone
    arch = cfg.MODEL.BACKBONE.ARCH
    # for head
    dropout_rate = cfg.MODEL.HEAD.DROPOUT_RATE
    num_classes = cfg.MODEL.HEAD.NUM_CLASSES

    return OfficialResNeSt(arch=arch,
                           dropout_rate=dropout_rate,
                           num_classes=num_classes,
                           fix_bn=fix_bn,
                           partial_bn=partial_bn,
                           pretrained=pretrained,
                           pretrained_num_classes=pretrained_num_classes
                           )

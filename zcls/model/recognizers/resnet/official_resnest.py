# -*- coding: utf-8 -*-

"""
@date: 2021/1/7 下午7:51
@file: official_resnest.py
@author: zj
@description: 
"""
from abc import ABC

import torch.nn as nn
from torch.nn.modules.module import T
from resnest.torch.resnest import resnest50, resnest101, resnest200, resnest269
from resnest.torch.resnet import ResNet, Bottleneck

from zcls.config.key_word import KEY_OUTPUT
from zcls.model import registry
from zcls.model.norm_helper import freezing_bn
from zcls.model.misc import load_pretrained_weights


class OfficialResNeSt(nn.Module, ABC):

    def __init__(self,
                 cfg,
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

        self._init_weights(cfg)

    def _init_weights(self, cfg):
        pretrained_local = cfg.MODEL.RECOGNIZER.PRETRAINED_LOCAL
        pretrained_num_classes = cfg.MODEL.RECOGNIZER.PRETRAINED_NUM_CLASSES
        num_classes = cfg.MODEL.HEAD.NUM_CLASSES
        load_pretrained_weights(self, cfg.MODEL.BACKBONE.ARCH,
                                weights_path=None if pretrained_local == '' else pretrained_local,
                                load_fc=pretrained_num_classes == num_classes,
                                verbose=True,
                                url_map=None
                                )

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
    pretrained = cfg.MODEL.RECOGNIZER.PRETRAINED_LOCAL
    pretrained_num_classes = cfg.MODEL.RECOGNIZER.PRETRAINED_NUM_CLASSES
    fix_bn = cfg.MODEL.NORM.FIX_BN
    partial_bn = cfg.MODEL.NORM.PARTIAL_BN
    # for backbone
    arch = cfg.MODEL.BACKBONE.ARCH
    # for head
    dropout_rate = cfg.MODEL.HEAD.DROPOUT_RATE
    num_classes = cfg.MODEL.HEAD.NUM_CLASSES

    return OfficialResNeSt(cfg,
                           arch=arch,
                           dropout_rate=dropout_rate,
                           num_classes=num_classes,
                           fix_bn=fix_bn,
                           partial_bn=partial_bn,
                           pretrained=pretrained,
                           pretrained_num_classes=pretrained_num_classes
                           )

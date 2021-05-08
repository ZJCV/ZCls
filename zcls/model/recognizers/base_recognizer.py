# -*- coding: utf-8 -*-

"""
@date: 2021/2/22 下午2:38
@file: base_recognizer.py
@author: zj
@description: 
"""

from abc import ABC

import torch
import torch.nn as nn
from torch.nn.modules.module import T

from zcls.config.key_word import KEY_OUTPUT
from zcls.model.backbones.build import build_backbone
from zcls.model.heads.build import build_head
from zcls.model.norm_helper import freezing_bn
from zcls.util.checkpoint import CheckPointer
from zcls.util import logging

logger = logging.get_logger(__name__)


class BaseRecognizer(nn.Module, ABC):

    def __init__(self, cfg):
        super(BaseRecognizer, self).__init__()
        self.fix_bn = cfg.MODEL.NORM.FIX_BN
        self.partial_bn = cfg.MODEL.NORM.PARTIAL_BN

        self.backbone = build_backbone(cfg)
        self.head = build_head(cfg)

        zcls_pretrained = cfg.MODEL.RECOGNIZER.PRETRAINED
        pretrained_num_classes = cfg.MODEL.RECOGNIZER.PRETRAINED_NUM_CLASSES
        num_classes = cfg.MODEL.HEAD.NUM_CLASSES
        self.init_weights(zcls_pretrained,
                          pretrained_num_classes,
                          num_classes)

    def init_weights(self, pretrained, pretrained_num_classes, num_classes):
        if pretrained != "":
            logger.info(f'load pretrained: {pretrained}')
            check_pointer = CheckPointer(model=self)
            check_pointer.load(pretrained, map_location=torch.device('cpu'))
            logger.info("finish loading model weights")
        if num_classes != pretrained_num_classes:
            fc = self.head.fc
            fc_features = fc.in_features

            fc = nn.Linear(fc_features, num_classes)
            nn.init.normal_(fc.weight, 0, 0.01)
            nn.init.zeros_(fc.bias)

            self.head.fc = fc

    def train(self, mode: bool = True) -> T:
        super(BaseRecognizer, self).train(mode=mode)

        if mode and (self.partial_bn or self.fix_bn):
            freezing_bn(self, partial_bn=self.partial_bn)

        return self

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)

        return {KEY_OUTPUT: x}

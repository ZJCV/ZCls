# -*- coding: utf-8 -*-

"""
@date: 2021/2/22 下午2:38
@file: base_recognizer.py
@author: zj
@description: 
"""

from abc import ABC

import torch.nn as nn
from torch.nn.modules.module import T

from zcls.config.key_word import KEY_OUTPUT
from zcls.model.backbones.build import build_backbone
from zcls.model.heads.build import build_head
from zcls.model.norm_helper import freezing_bn
from zcls.util import logging
from ..misc import load_pretrained_weights

logger = logging.get_logger(__name__)


class BaseRecognizer(nn.Module, ABC):

    def __init__(self, cfg):
        super(BaseRecognizer, self).__init__()
        self.fix_bn = cfg.MODEL.NORM.FIX_BN
        self.partial_bn = cfg.MODEL.NORM.PARTIAL_BN

        self.backbone = build_backbone(cfg)
        self.head = build_head(cfg)

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
        super(BaseRecognizer, self).train(mode=mode)

        if mode and (self.partial_bn or self.fix_bn):
            freezing_bn(self, partial_bn=self.partial_bn)

        return self

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)

        return {KEY_OUTPUT: x}

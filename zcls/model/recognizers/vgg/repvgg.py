# -*- coding: utf-8 -*-

"""
@date: 2021/2/2 下午5:19
@file: repvgg.py
@author: zj
@description: RegVGG，参考[RepVGG/repvgg.py](https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py)
"""

import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url

from zcls.config.key_word import KEY_OUTPUT
from zcls.model import registry
from zcls.model.backbones.build import build_backbone
from zcls.model.heads.build import build_head


@registry.RECOGNIZER.register('RepVGG')
class RepVGG(nn.Module):

    def __init__(self, cfg):
        super(RepVGG, self).__init__()

        self.backbone = build_backbone(cfg)
        self.head = build_head(cfg)

        zcls_pretrained = cfg.MODEL.RECOGNIZER.PRETRAINED
        pretrained_num_classes = cfg.MODEL.RECOGNIZER.PRETRAINED_NUM_CLASSES
        num_classes = cfg.MODEL.HEAD.NUM_CLASSES
        self.init_weights(zcls_pretrained,
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

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)

        return {KEY_OUTPUT: x}

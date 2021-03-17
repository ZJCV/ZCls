# -*- coding: utf-8 -*-

"""
@date: 2020/10/19 下午2:38
@file: crossentropy_loss.py
@author: zj
@description: 
"""

from abc import ABC
import torch.nn as nn

from zcls.config.key_word import KEY_OUTPUT, KEY_LOSS
from .. import registry


@registry.CRITERION.register('CrossEntropyLoss')
class CrossEntropyLoss(nn.Module, ABC):

    def __init__(self, cfg):
        super(CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction='mean')

    def __call__(self, output_dict, targets):
        assert isinstance(output_dict, dict) and KEY_OUTPUT in output_dict.keys()
        inputs = output_dict[KEY_OUTPUT]
        loss = self.loss(inputs, targets)

        return {KEY_LOSS: loss}

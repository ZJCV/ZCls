# -*- coding: utf-8 -*-

"""
@date: 2020/11/12 下午5:49
@file: resnet_pytorch.py
@author: zj
@description: 
"""
from typing import Any

import torch.nn as nn
import torchvision.models.resnet as resnet

class ResNet_Pytorch(nn.Module):

    def __init__(self, cfg):
        super(ResNet_Pytorch, self).__init__()
        type = cfg.MODEL.RECOGNIZER.NAME

        if type == 'R50_Pytorch':
            model = resnet.resnet50()
        else:
            raise ValueError(f'{type} does not exist')

        self.model = model

    def forward(self, *input: Any, **kwargs: Any):
        probs = self.model.forward(*input, **kwargs)
        return {'probs': probs}

# -*- coding: utf-8 -*-

"""
@date: 2020/12/14 下午6:56
@file: squeeze_and_excitation_block.py
@author: zj
@description: 
"""

import torch.nn as nn

from ..act_helper import get_sigmoid
from ..backbones.misc import round_to_multiple_of


class _SqueezeAndExcitationBlockND(nn.Module):

    def __init__(self,
                 in_channels,
                 reduction=16,
                 dimension=2,
                 sigmoid_type='Sigmoid',
                 bias=False,
                 is_round=False,
                 round_nearest=8,
                 ):
        """
        Squeeze-and-Excitation Block
        refer to
        [se_module.py](https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py)
        [如何评价Momenta ImageNet 2017夺冠架构SENet?](https://www.zhihu.com/question/63460684)
        :param in_channels:
        :param reduction:
        :param dimension:
        :param sigmoid_type:
        """
        super(_SqueezeAndExcitationBlockND, self).__init__()
        assert dimension in [1, 2, 3]
        assert in_channels % reduction == 0, f'in_channels = {in_channels}, reduction = {reduction}'

        inner_channel = in_channels // reduction
        if is_round:
            inner_channel = round_to_multiple_of(inner_channel, round_nearest)
        if dimension == 1:
            self.squeeze = nn.AdaptiveAvgPool1d((1))
        elif dimension == 2:
            self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.squeeze = nn.AdaptiveAvgPool3d((1, 1, 1))

        sigmoid_layer = get_sigmoid(sigmoid_type)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, inner_channel, bias=bias),
            nn.ReLU(inplace=True),
            nn.Linear(inner_channel, in_channels, bias=bias),
            sigmoid_layer()
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

    def forward(self, x):
        """
        :param x: (N, C, **)
        """
        self._check_input_dim(x)
        N, C = x.shape[:2]

        out = self.squeeze(x)
        out_shape = out.shape
        out = self.excitation(out.view(N, C)).view(out_shape)
        scale = x * out.expand_as(x)
        return scale

    def _check_input_dim(self, input):
        raise NotImplementedError


class SqueezeAndExcitationBlock1D(_SqueezeAndExcitationBlockND):

    def __init__(self, in_channels, reduction=16, dimension=1, **kwargs):
        super().__init__(in_channels, reduction, dimension, **kwargs)

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))


class SqueezeAndExcitationBlock2D(_SqueezeAndExcitationBlockND):

    def __init__(self, in_channels, reduction=16, dimension=2, **kwargs):
        super().__init__(in_channels, reduction, dimension, **kwargs)

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


class SqueezeAndExcitationBlock3D(_SqueezeAndExcitationBlockND):

    def __init__(self, in_channels, reduction=16, dimension=3, **kwargs):
        super().__init__(in_channels, reduction, dimension, **kwargs)

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))

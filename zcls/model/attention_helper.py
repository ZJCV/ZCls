# -*- coding: utf-8 -*-

"""
@date: 2020/12/30 下午4:44
@file: attention_helper.py
@author: zj
@description: 
"""

from zcls.model.layers.global_context_block import GlobalContextBlock2D
from zcls.model.layers.squeeze_and_excitation_block import SqueezeAndExcitationBlock2D
from zcls.model.layers.non_local_embedded_gaussian import NonLocal2DEmbeddedGaussian
from zcls.model.layers.simplified_non_local_embedded_gaussian import SimplifiedNonLocal2DEmbeddedGaussian


def make_attention_block(in_planes, reduction, attention_type, **kwargs):
    if attention_type == 'GlobalContextBlock2D':
        return GlobalContextBlock2D(in_channels=in_planes, reduction=reduction)
    elif attention_type == 'SqueezeAndExcitationBlock2D':
        return SqueezeAndExcitationBlock2D(in_channels=in_planes, reduction=reduction, **kwargs)
    elif attention_type == 'NonLocal2DEmbeddedGaussian':
        return NonLocal2DEmbeddedGaussian(in_channels=in_planes)
    elif attention_type == 'SimplifiedNonLocal2DEmbeddedGaussian':
        return SimplifiedNonLocal2DEmbeddedGaussian(in_channels=in_planes)
    else:
        raise ValueError('no matching type')

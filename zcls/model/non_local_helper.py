# -*- coding: utf-8 -*-

"""
@date: 2020/12/8 下午5:17
@file: non_local_helper.py
@author: zj
@description: 
"""

import torch.nn as nn

from .layers.non_local_embedded_gaussian import NonLocal2DEmbeddedGaussian


class NL2DWrapper(nn.Module):

    def __init__(self, block, nl_type='EmbeddedGaussian'):
        super(NL2DWrapper, self).__init__()
        if nl_type == 'EmbeddedGaussian':
            NLBlock = NonLocal2DEmbeddedGaussian
        else:
            raise NotImplementedError

        self.block = block
        self.nl = NLBlock(block.bn3.num_features)

    def forward(self, x):
        x = self.block(x)
        x = self.nl(x)
        return x


def make_non_local_2d(net, net_type='ResNet_Pytorch', arch_type='resnet50', nl_type='EmbeddedGaussian'):
    if net_type == 'ResNet_Pytorch' and arch_type == 'resnet50':
        net.layer2 = nn.Sequential(
            NL2DWrapper(net.layer2[0], nl_type=nl_type),
            net.layer2[1],
            NL2DWrapper(net.layer2[2]),
            net.layer2[3],
        )
        net.layer3 = nn.Sequential(
            NL2DWrapper(net.layer3[0]),
            net.layer3[1],
            NL2DWrapper(net.layer3[2]),
            net.layer3[3],
            NL2DWrapper(net.layer3[4]),
            net.layer3[5],
        )
    elif net_type == 'ResNet_Custom' and arch_type == 'resnet50':
        net = net.backbone
        net.layer2 = nn.Sequential(
            NL2DWrapper(net.layer2[0]),
            net.layer2[1],
            NL2DWrapper(net.layer2[2]),
            net.layer2[3],
        )
        net.layer3 = nn.Sequential(
            NL2DWrapper(net.layer3[0]),
            net.layer3[1],
            NL2DWrapper(net.layer3[2]),
            net.layer3[3],
            NL2DWrapper(net.layer3[4]),
            net.layer3[5],
        )
    else:
        raise NotImplementedError

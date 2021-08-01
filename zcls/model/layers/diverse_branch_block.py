# -*- coding: utf-8 -*-

"""
@date: 2021/7/28 下午5:51
@file: diver_branch_block.py
@author: zj
@description:
refer to [DingXiaoH/DiverseBranchBlock](https://github.com/DingXiaoH/DiverseBranchBlock)
"""

import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .. import init_helper


def conv_bn(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
            padding_mode='zeros'):
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                           stride=(stride, stride), padding=padding, dilation=(dilation, dilation), groups=groups,
                           bias=False, padding_mode=padding_mode)
    bn_layer = nn.BatchNorm2d(num_features=out_channels, affine=True)
    se = nn.Sequential()
    se.add_module('conv', conv_layer)
    se.add_module('bn', bn_layer)
    return se


class IdentityBasedConv1x1(nn.Conv2d):

    def __init__(self, channels, groups=1):
        super(IdentityBasedConv1x1, self).__init__(in_channels=channels, out_channels=channels,
                                                   kernel_size=(1, 1), stride=(1, 1),
                                                   padding=0, groups=groups, bias=False)

        assert channels % groups == 0
        input_dim = channels // groups
        id_value = np.zeros((channels, input_dim, 1, 1))
        for i in range(channels):
            id_value[i, i % input_dim, 0, 0] = 1
        self.id_tensor = torch.from_numpy(id_value).type_as(self.weight)
        nn.init.zeros_(self.weight)

    def forward(self, input):
        kernel = self.weight + self.id_tensor.to(self.weight.device)
        result = F.conv2d(input, kernel, None, stride=1, padding=0, dilation=self.dilation, groups=self.groups)
        return result

    def get_actual_kernel(self):
        return self.weight + self.id_tensor.to(self.weight.device)


class BNAndPadLayer(nn.Module):
    def __init__(self,
                 pad_pixels,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = self.bn.bias.detach() - self.bn.running_mean * self.bn.weight.detach() / torch.sqrt(
                    self.bn.running_var + self.bn.eps)
            else:
                pad_values = - self.bn.running_mean / torch.sqrt(self.bn.running_var + self.bn.eps)
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0:self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels:, :] = pad_values
            output[:, :, :, 0:self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels:] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps


class DiverseBranchBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 internal_channels_1x1_3x3=None,
                 single_init=False):
        super(DiverseBranchBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        assert padding == kernel_size // 2

        self.dbb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, dilation=dilation, groups=groups)

        self.dbb_avg = nn.Sequential()
        if groups < out_channels:
            self.dbb_avg.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                      kernel_size=(1, 1), stride=(1, 1),
                                                      padding=0, groups=groups, bias=False))
            self.dbb_avg.add_module('bn', BNAndPadLayer(pad_pixels=padding, num_features=out_channels))
            self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))

            # For ordinary convolution, an additional 1x1conv+BN branch is added
            self.dbb_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   padding=0, groups=groups)
        else:
            # For depthwise convolution, the pre 1x1conv+BN are cancelled
            self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding))

        self.dbb_avg.add_module('avgbn', nn.BatchNorm2d(out_channels))

        if internal_channels_1x1_3x3 is None:
            # For mobilenet, it is better to have 2X internal channels
            internal_channels_1x1_3x3 = in_channels if groups < out_channels else 2 * in_channels

        self.dbb_1x1_kxk = nn.Sequential()
        if internal_channels_1x1_3x3 == in_channels:
            self.dbb_1x1_kxk.add_module('idconv1', IdentityBasedConv1x1(channels=in_channels, groups=groups))
        else:
            self.dbb_1x1_kxk.add_module('conv1', nn.Conv2d(in_channels=in_channels,
                                                           out_channels=internal_channels_1x1_3x3,
                                                           kernel_size=(1, 1), stride=(1, 1), padding=0, groups=groups,
                                                           bias=False))
        self.dbb_1x1_kxk.add_module('bn1', BNAndPadLayer(pad_pixels=padding, num_features=internal_channels_1x1_3x3,
                                                         affine=True))
        self.dbb_1x1_kxk.add_module('conv2',
                                    nn.Conv2d(in_channels=internal_channels_1x1_3x3, out_channels=out_channels,
                                              kernel_size=kernel_size, stride=(stride, stride), padding=0,
                                              groups=groups, bias=False))
        self.dbb_1x1_kxk.add_module('bn2', nn.BatchNorm2d(out_channels))

        self.init_weights()
        #   The experiments reported in the paper used the default initialization of bn.weight (all as 1). But changing the initialization may be useful in some cases.
        if single_init:
            #   Initialize the bn.weight of dbb_origin as 1 and others as 0. This is not the default setting.
            self.single_init()

    def forward(self, inputs):
        out = self.dbb_origin(inputs)
        if hasattr(self, 'dbb_1x1'):
            out += self.dbb_1x1(inputs)
        out += self.dbb_avg(inputs)
        out += self.dbb_1x1_kxk(inputs)
        return out

    def init_weights(self):
        if hasattr(self, "dbb_origin"):
            init_helper.init_weights(self.dbb_origin)
        if hasattr(self, "dbb_1x1"):
            init_helper.init_weights(self.dbb_1x1)
        if hasattr(self, "dbb_avg"):
            init_helper.init_weights(self.dbb_avg)
        if hasattr(self, "dbb_1x1_kxk"):
            init_helper.init_weights(self.dbb_1x1_kxk)

    def init_gamma(self, gamma_value):
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, gamma_value)
        if hasattr(self, "dbb_1x1"):
            torch.nn.init.constant_(self.dbb_1x1.bn.weight, gamma_value)
        if hasattr(self, "dbb_avg"):
            torch.nn.init.constant_(self.dbb_avg.avgbn.weight, gamma_value)
        if hasattr(self, "dbb_1x1_kxk"):
            torch.nn.init.constant_(self.dbb_1x1_kxk.bn2.weight, gamma_value)

    def single_init(self):
        self.init_gamma(0.0)
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, 1.0)

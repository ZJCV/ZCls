# -*- coding: utf-8 -*-

"""
@date: 2021/2/1 下午7:10
@file: asymmetric_convolution_block.py
@author: zj
@description: 
"""

import torch.nn as nn


class AsymmetricConvolutionBlock(nn.Module):
    """
    refer to [ACNet/acnet/acb.py](https://github.com/DingXiaoH/ACNet/blob/master/acnet/acb.py)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 padding_mode='zeros', use_affine=True, use_last_bn=False):
        super(AsymmetricConvolutionBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=(kernel_size, kernel_size), stride=stride,
                                     padding=padding, dilation=dilation, groups=groups, bias=False,
                                     padding_mode=padding_mode)
        self.square_bn = nn.BatchNorm2d(num_features=out_channels, affine=use_affine)

        center_offset_from_origin_border = padding - kernel_size // 2
        ver_pad_or_crop = (padding, center_offset_from_origin_border)
        hor_pad_or_crop = (center_offset_from_origin_border, padding)
        if center_offset_from_origin_border >= 0:
            # padding
            self.ver_conv_crop_layer = nn.Identity()
            ver_conv_padding = ver_pad_or_crop
            self.hor_conv_crop_layer = nn.Identity()
            hor_conv_padding = hor_pad_or_crop
        else:
            # crop
            self.ver_conv_crop_layer = CropLayer(crop_set=ver_pad_or_crop)
            ver_conv_padding = (0, 0)
            self.hor_conv_crop_layer = CropLayer(crop_set=hor_pad_or_crop)
            hor_conv_padding = (0, 0)
        self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, 1),
                                  stride=stride,
                                  padding=ver_conv_padding, dilation=dilation, groups=groups, bias=False,
                                  padding_mode=padding_mode)

        self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                                  stride=stride,
                                  padding=hor_conv_padding, dilation=dilation, groups=groups, bias=False,
                                  padding_mode=padding_mode)
        self.ver_bn = nn.BatchNorm2d(num_features=out_channels, affine=use_affine)
        self.hor_bn = nn.BatchNorm2d(num_features=out_channels, affine=use_affine)

        if use_last_bn:
            self.last_bn = nn.BatchNorm2d(num_features=out_channels, affine=True)

        self._init_weights()

    def _init_weights(self, gamma=1.0 / 3):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, gamma)
                nn.init.constant_(m.bias, gamma)

    def forward(self, input):
        square_outputs = self.square_conv(input)
        square_outputs = self.square_bn(square_outputs)
        vertical_outputs = self.ver_conv_crop_layer(input)
        vertical_outputs = self.ver_conv(vertical_outputs)
        vertical_outputs = self.ver_bn(vertical_outputs)
        horizontal_outputs = self.hor_conv_crop_layer(input)
        horizontal_outputs = self.hor_conv(horizontal_outputs)
        horizontal_outputs = self.hor_bn(horizontal_outputs)
        result = square_outputs + vertical_outputs + horizontal_outputs
        if hasattr(self, 'last_bn'):
            return self.last_bn(result)
        return result


class CropLayer(nn.Module):

    #   E.g., (-1, 0) means this layer should crop the first and last rows of the feature map. And (0, -1) crops the first and last columns
    def __init__(self, crop_set):
        super(CropLayer, self).__init__()
        self.rows_to_crop = - crop_set[0]
        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0

    def forward(self, input):
        if self.rows_to_crop == 0 and self.cols_to_crop == 0:
            return input
        elif self.rows_to_crop > 0 and self.cols_to_crop == 0:
            return input[:, :, self.rows_to_crop:-self.rows_to_crop, :]
        elif self.rows_to_crop == 0 and self.cols_to_crop > 0:
            return input[:, :, :, self.cols_to_crop:-self.cols_to_crop]
        else:
            return input[:, :, self.rows_to_crop:-self.rows_to_crop, self.cols_to_crop:-self.cols_to_crop]

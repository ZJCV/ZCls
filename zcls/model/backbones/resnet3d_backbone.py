# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午7:04
@file: resnet_backbone.py
@author: zj
@description: 
"""

import torch.nn as nn

from ..backbones.resnet3d_basicblock import ResNet3DBasicBlock
from ..backbones.resnet3d_bottleneck import ResNet3DBottleneck


class ResNet3DBackbone(nn.Module):

    def __init__(self,
                 # 输入通道数
                 inplanes=3,
                 # 基础通道数,
                 base_planes=64,
                 # 第一个卷积层kernel_size
                 conv1_kernel=(1, 7, 7),
                 # 第一个卷积层步长
                 conv1_stride=(1, 2, 2),
                 # 第一个卷积层零填充
                 conv1_padding=(0, 3, 3),
                 # 第一个池化层大小
                 pool1_kernel=(1, 3, 3),
                 # 第一个池化层步长
                 pool1_stride=(1, 2, 2),
                 # 第一个池化层零填充
                 pool1_padding=(0, 1, 1),
                 # 是否使用第二个池化层
                 with_pool2=False,
                 # 每一层通道数
                 layer_planes=(64, 128, 256, 512),
                 # 每一层块个数
                 layer_blocks=(2, 2, 2, 2),
                 # 是否执行空间下采样，0表示不执行，1表示执行
                 downsamples=(0, 1, 1, 1),
                 # 时间步长
                 temporal_strides=(1, 1, 1, 1),
                 # 是否执行膨胀操作，0表示不执行，1表示执行
                 inflate_list=(0, 0, 0, 0),
                 # 膨胀类型
                 inflate_style='3x1x1',
                 # 块类型
                 block_layer=None,
                 # 卷积层类型
                 conv_layer=None,
                 # 归一化层类型
                 norm_layer=None,
                 # 激活层类型
                 act_layer=None,
                 # 零初始化残差连接
                 zero_init_residual=False,
                 # 2d预训练模型
                 state_dict_2d=None,
                 ):
        super(ResNet3DBackbone, self).__init__()

        if block_layer is None:
            block_layer = ResNet3DBasicBlock
        if conv_layer is None:
            conv_layer = nn.Conv3d
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if act_layer is None:
            act_layer = nn.ReLU

        self.inplanes = inplanes
        self.base_planes = base_planes
        self.conv1_kernel = conv1_kernel
        self.conv1_stride = conv1_stride
        self.conv1_padding = conv1_padding
        self.pool1_kernel = pool1_kernel
        self.pool1_stride = pool1_stride
        self.pool1_padding = pool1_padding
        self.with_pool2 = with_pool2
        self.layer_planes = layer_planes
        self.layer_blocks = layer_blocks
        self.downsamples = downsamples
        self.temporal_strides = temporal_strides
        self.inflate_list = inflate_list
        self.inflate_style = inflate_style
        self.block_layer = block_layer
        self.conv_layer = conv_layer
        self.norm_layer = norm_layer
        self.act_layer = act_layer

        self._make_stem()
        self.inplanes = self.base_planes
        for i in range(len(self.layer_blocks)):
            res_layer = self._make_res_layer(self.inplanes,
                                             self.layer_planes[i],
                                             self.layer_blocks[i],
                                             self.downsamples[i],
                                             self.temporal_strides[i],
                                             self.inflate_list[i],
                                             self.inflate_style,
                                             self.block_layer,
                                             self.conv_layer,
                                             self.norm_layer,
                                             self.act_layer
                                             )
            self.inplanes = self.layer_planes[i] * self.block_layer.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)

        self._init_weights(zero_init_residual, state_dict_2d)

    def _make_stem(self):
        self.conv1 = self.conv_layer(self.inplanes, self.base_planes,
                                     kernel_size=self.conv1_kernel, stride=self.conv1_stride,
                                     padding=self.conv1_padding, bias=False)
        self.bn1 = self.norm_layer(self.base_planes)
        self.relu = self.act_layer(inplace=True)

        self.maxpool = nn.MaxPool3d(kernel_size=self.pool1_kernel, stride=self.pool1_stride, padding=self.pool1_padding)

        if self.with_pool2:
            self.maxpool2 = nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))

    def _make_res_layer(self,
                        # 输入通道数
                        inplanes,
                        # 输出通道数
                        planes,
                        # 块个数
                        block_num,
                        # 是否执行空间下采样
                        is_downsample,
                        # 时间步长
                        temporal_stride,
                        # 是否执行膨胀操作
                        inflate,
                        # 膨胀类型
                        inflate_style,
                        # 块类型
                        block_layer,
                        # 卷积层类型
                        conv_layer,
                        # 归一化层类型
                        norm_layer,
                        # 激活层类型
                        act_layer,
                        ):
        inflates = inflate if not isinstance(inflate, int) else (inflate,) * block_num
        assert len(inflates) == block_num
        assert inflate_style in ('3x1x1', '3x3x3')

        spatial_stride = 2 if is_downsample else 1
        expansion = self.block_layer.expansion
        if is_downsample or inplanes != planes * expansion:
            downsample = nn.Sequential(
                conv_layer(inplanes, planes * expansion, kernel_size=1,
                           stride=(temporal_stride, spatial_stride, spatial_stride), bias=False),
                norm_layer(planes * expansion),
            )
        else:
            downsample = None

        blocks = list()
        # 仅对第一个block执行可能的时间或者空间下采样
        blocks.append(block_layer(inplanes, planes,
                                  spatial_stride, temporal_stride, downsample,
                                  inflates[0], inflate_style,
                                  conv_layer, norm_layer, act_layer))
        inplanes = planes * expansion

        for i in range(1, block_num):
            blocks.append(block_layer(inplanes, planes,
                                      1, 1, None,
                                      inflates[i], inflate_style,
                                      conv_layer, norm_layer, act_layer))
        return nn.Sequential(*blocks)

    def _init_weights(self, zero_init_residual, state_dict_2d):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResNet3DBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, ResNet3DBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        if state_dict_2d:

            def _inflate_conv_params(conv3d, state_dict_2d, module_name_2d,
                                     inflated_param_names):
                """Inflate a conv module from 2d to 3d.

                Args:
                    conv3d (nn.Module): The destination conv3d module.
                    state_dict_2d (OrderedDict): The state dict of pretrained 2d model.
                    module_name_2d (str): The name of corresponding conv module in the
                        2d model.
                    inflated_param_names (list[str]): List of parameters that have been
                        inflated.
                """
                weight_2d_name = module_name_2d + '.weight'

                conv2d_weight = state_dict_2d[weight_2d_name]
                kernel_t = conv3d.weight.data.shape[2]

                new_weight = conv2d_weight.data.unsqueeze(2).expand_as(
                    conv3d.weight) / kernel_t
                conv3d.weight.data.copy_(new_weight)
                inflated_param_names.append(weight_2d_name)

                if getattr(conv3d, 'bias') is not None:
                    bias_2d_name = module_name_2d + '.bias'
                    conv3d.bias.data.copy_(state_dict_2d[bias_2d_name])
                    inflated_param_names.append(bias_2d_name)

            def _inflate_bn_params(bn3d, state_dict_2d, module_name_2d,
                                   inflated_param_names):
                """Inflate a norm module from 2d to 3d.

                Args:
                    bn3d (nn.Module): The destination bn3d module.
                    state_dict_2d (OrderedDict): The state dict of pretrained 2d model.
                    module_name_2d (str): The name of corresponding bn module in the
                        2d model.
                    inflated_param_names (list[str]): List of parameters that have been
                        inflated.
                """
                for param_name, param in bn3d.named_parameters():
                    param_2d_name = f'{module_name_2d}.{param_name}'
                    param_2d = state_dict_2d[param_2d_name]
                    param.data.copy_(param_2d)
                    inflated_param_names.append(param_2d_name)

                for param_name, param in bn3d.named_buffers():
                    param_2d_name = f'{module_name_2d}.{param_name}'
                    # some buffers like num_batches_tracked may not exist in old
                    # checkpoints
                    if param_2d_name in state_dict_2d:
                        param_2d = state_dict_2d[param_2d_name]
                        param.data.copy_(param_2d)
                        inflated_param_names.append(param_2d_name)

            inflated_param_names = []
            for name, module in self.named_modules():
                if isinstance(module, nn.Conv3d):
                    _inflate_conv_params(module, state_dict_2d, name, inflated_param_names)
                if isinstance(module, (nn.BatchNorm3d, nn.GroupNorm)):
                    _inflate_bn_params(module, state_dict_2d, name, inflated_param_names)

            # check if any parameters in the 2d checkpoint are not loaded
            remaining_names = set(
                state_dict_2d.keys()) - set(inflated_param_names)
            if remaining_names:
                print(f'These parameters in the 2d checkpoint are not loaded: {sorted(remaining_names)}')

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if self.with_pool2:
            x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

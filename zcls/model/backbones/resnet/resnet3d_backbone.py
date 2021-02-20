# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午7:04
@file: resnet_backbone.py
@author: zj
@description: 
"""

from abc import ABC
import torch.nn as nn
from torchvision.models.resnet import model_urls
from torchvision.models.utils import load_state_dict_from_url

from zcls.model import registry
from zcls.model.conv_helper import get_conv
from zcls.model.norm_helper import get_norm
from zcls.model.act_helper import get_act
from zcls.model.layers.place_holder import PlaceHolder
from .resnet3d_basicblock import ResNet3DBasicBlock
from .resnet3d_bottleneck import ResNet3DBottleneck

arch_settings = {
    'resnet18': (ResNet3DBasicBlock, (2, 2, 2, 2), 1, 64),
    'resnet34': (ResNet3DBasicBlock, (3, 4, 6, 3), 1, 64),
    'resnet50': (ResNet3DBottleneck, (3, 4, 6, 3), 1, 64),
    'resnet101': (ResNet3DBottleneck, (3, 4, 23, 3), 1, 64),
    'resnet152': (ResNet3DBottleneck, (3, 8, 36, 3), 1, 64),
    'resnext50_32x4d': (ResNet3DBottleneck, (3, 4, 6, 3), 32, 4),
    'resnext101_32x8d': (ResNet3DBottleneck, (3, 4, 23, 3), 32, 8)
}


class ResNet3DBackbone(nn.Module, ABC):

    def __init__(self,
                 # 输入通道数
                 in_planes=3,
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
                 down_samples=(0, 1, 1, 1),
                 # 时间步长
                 temporal_strides=(1, 1, 1, 1),
                 # 是否执行膨胀操作，0表示不执行，1表示执行
                 inflate_list=(0, 0, 0, 0),
                 # 膨胀类型
                 inflate_style='3x1x1',
                 # cardinality
                 groups=1,
                 # 每组的宽度
                 width_per_group=64,
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

        self._make_stem(in_planes,
                        base_planes,
                        conv1_kernel,
                        conv1_stride,
                        conv1_padding,
                        pool1_kernel,
                        pool1_stride,
                        pool1_padding,
                        with_pool2,
                        conv_layer,
                        norm_layer,
                        act_layer
                        )
        inplanes = base_planes
        for i in range(len(layer_blocks)):
            res_layer = self._make_res_layer(inplanes,
                                             layer_planes[i],
                                             layer_blocks[i],
                                             down_samples[i],
                                             temporal_strides[i],
                                             inflate_list[i],
                                             inflate_style,
                                             groups,
                                             width_per_group,
                                             block_layer,
                                             conv_layer,
                                             norm_layer,
                                             act_layer
                                             )
            inplanes = layer_planes[i] * block_layer.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)

        self.init_weights(zero_init_residual, state_dict_2d)

    def _make_stem(self,
                   in_planes,
                   base_planes,
                   conv1_kernel,
                   conv1_stride,
                   conv1_padding,
                   pool1_kernel,
                   pool1_stride,
                   pool1_padding,
                   with_pool2,
                   conv_layer,
                   norm_layer,
                   act_layer,
                   ):
        self.conv1 = conv_layer(in_planes, base_planes,
                                kernel_size=conv1_kernel, stride=conv1_stride,
                                padding=conv1_padding, bias=False)
        self.bn1 = norm_layer(base_planes)
        self.relu = act_layer(inplace=True)

        self.pool = nn.MaxPool3d(kernel_size=pool1_kernel, stride=pool1_stride, padding=pool1_padding)

        if with_pool2:
            self.pool2 = nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        else:
            self.pool2 = PlaceHolder()

    def _make_res_layer(self,
                        # 输入通道数
                        in_planes,
                        # 输出通道数
                        out_planes,
                        # 块个数
                        block_num,
                        # 是否执行空间下采样
                        with_down_sample,
                        # 时间步长
                        temporal_stride,
                        # 是否执行膨胀操作
                        inflate,
                        # 膨胀类型
                        inflate_style,
                        # cardinality
                        groups,
                        # 每组的宽度
                        width_per_group,
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

        spatial_stride = 2 if with_down_sample else 1
        expansion = block_layer.expansion
        if with_down_sample or in_planes != out_planes * expansion:
            down_sample = nn.Sequential(
                conv_layer(in_planes, out_planes * expansion, kernel_size=1,
                           stride=(temporal_stride, spatial_stride, spatial_stride), bias=False),
                norm_layer(out_planes * expansion),
            )
        else:
            down_sample = None

        blocks = list()
        # 仅对第一个block执行可能的时间或者空间下采样
        blocks.append(block_layer(in_planes, out_planes,
                                  spatial_stride, temporal_stride, down_sample,
                                  inflates[0], inflate_style,
                                  groups, width_per_group,
                                  conv_layer, norm_layer, act_layer))
        in_planes = out_planes * expansion

        spatial_stride = 1
        temporal_stride = 1
        down_sample = None
        for i in range(1, block_num):
            blocks.append(block_layer(in_planes, out_planes,
                                      spatial_stride, temporal_stride, down_sample,
                                      inflates[i], inflate_style,
                                      groups, width_per_group,
                                      conv_layer, norm_layer, act_layer))
        return nn.Sequential(*blocks)

    def init_weights(self, zero_init_residual, state_dict_2d):
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
                if weight_2d_name in state_dict_2d.keys():
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
                    if param_2d_name in state_dict_2d.keys():
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
        x = self.pool(x)

        x = self.layer1(x)
        x = self.pool2(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


@registry.Backbone.register('ResNet3D')
def build_resnet3d_backbone(cfg):
    arch = cfg.MODEL.BACKBONE.ARCH
    torchvision_pretrained = cfg.MODEL.RECOGNIZER.TORCHVISION_PRETRAINED
    conv_layer = get_conv(cfg)
    norm_layer = get_norm(cfg)
    act_layer = get_act(cfg)
    zero_init_residual = cfg.MODEL.RECOGNIZER.ZERO_INIT_RESIDUAL
    # for backbone
    in_planes = cfg.MODEL.BACKBONE.IN_PLANES
    base_planes = cfg.MODEL.BACKBONE.BASE_PLANES
    conv1_kernel = cfg.MODEL.BACKBONE.CONV1_KERNEL
    conv1_stride = cfg.MODEL.BACKBONE.CONV1_STRIDE
    conv1_padding = cfg.MODEL.BACKBONE.CONV1_PADDING
    pool1_kernel = cfg.MODEL.BACKBONE.POOL1_KERNEL
    pool1_stride = cfg.MODEL.BACKBONE.POOL1_STRIDE
    pool1_padding = cfg.MODEL.BACKBONE.POOL1_PADDING
    with_pool2 = cfg.MODEL.BACKBONE.WITH_POOL2
    layer_planes = cfg.MODEL.BACKBONE.LAYER_PLANES
    down_samples = cfg.MODEL.BACKBONE.DOWNSAMPLES
    temporal_strides = cfg.MODEL.BACKBONE.TEMPORAL_STRIDES
    inflate_list = cfg.MODEL.BACKBONE.INFLATE_LIST
    inflate_style = cfg.MODEL.BACKBONE.INFLATE_STYLE

    block_layer, layer_blocks, groups, width_per_group = arch_settings[arch]
    state_dict_2d = load_state_dict_from_url(model_urls[arch], progress=True) \
        if torchvision_pretrained else None

    return ResNet3DBackbone(
        in_planes=in_planes,
        base_planes=base_planes,
        conv1_kernel=conv1_kernel,
        conv1_stride=conv1_stride,
        conv1_padding=conv1_padding,
        pool1_kernel=pool1_kernel,
        pool1_stride=pool1_stride,
        pool1_padding=pool1_padding,
        with_pool2=with_pool2,
        layer_planes=layer_planes,
        layer_blocks=layer_blocks,
        down_samples=down_samples,
        temporal_strides=temporal_strides,
        inflate_list=inflate_list,
        inflate_style=inflate_style,
        groups=groups,
        width_per_group=width_per_group,
        block_layer=block_layer,
        conv_layer=conv_layer,
        norm_layer=norm_layer,
        act_layer=act_layer,
        zero_init_residual=zero_init_residual,
        state_dict_2d=state_dict_2d
    )

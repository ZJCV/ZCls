# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午2:37
@file: resnet_recognizer.py
@author: zj
@description: 
"""
from abc import ABC

import torch.nn as nn
from torch.nn.modules.module import T
from torchvision.models.resnet import model_urls
from torchvision.models.utils import load_state_dict_from_url

from zcls.config.key_word import KEY_OUTPUT
from zcls.model import registry
from zcls.model.backbones.resnet.resnet3d_basicblock import ResNet3DBasicBlock
from zcls.model.backbones.resnet.resnet3d_bottleneck import ResNet3DBottleneck
from zcls.model.backbones.resnet.resnet3d_backbone import ResNet3DBackbone
from zcls.model.heads.resnet3d_head import ResNet3DHead
from zcls.model.norm_helper import get_norm, freezing_bn
from zcls.model.conv_helper import get_conv
from zcls.model.act_helper import get_act

arch_settings = {
    'resnet18': (ResNet3DBasicBlock, (2, 2, 2, 2), 1, 64),
    'resnet34': (ResNet3DBasicBlock, (3, 4, 6, 3), 1, 64),
    'resnet50': (ResNet3DBottleneck, (3, 4, 6, 3), 1, 64),
    'resnet101': (ResNet3DBottleneck, (3, 4, 23, 3), 1, 64),
    'resnet152': (ResNet3DBottleneck, (3, 8, 36, 3), 1, 64),
    'resnext50_32x4d': (ResNet3DBottleneck, (3, 4, 6, 3), 32, 4),
    'resnext101_32x8d': (ResNet3DBottleneck, (3, 4, 23, 3), 32, 8)
}


class ResNet3DRecognizer(nn.Module, ABC):

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
                 # 是否执行空间下采样，0表示不执行，1表示执行
                 down_samples=(0, 1, 1, 1),
                 # 时间步长
                 temporal_strides=(1, 1, 1, 1),
                 # 是否执行膨胀操作，0表示不执行，1表示执行
                 inflate_list=(0, 0, 0, 0),
                 # 膨胀类型
                 inflate_style='3x1x1',
                 # 卷积层类型
                 conv_layer=None,
                 # 归一化层类型
                 norm_layer=None,
                 # 激活层类型
                 act_layer=None,
                 # 零初始化残差连接
                 zero_init_residual=False,
                 ########### for head
                 # 指定架构
                 arch='resnet18',
                 # 随机失活概率
                 dropout_rate=0.,
                 # 输出类别数
                 num_classes=1000,
                 ############ for recognizer
                 # zcls预训练模型
                 pretrained="",
                 # torchvision预训练模型
                 torchvision_pretrained=False,
                 # 预训练模型类别数
                 pretrained_num_classes=1000,
                 # 固定BN
                 fix_bn=False,
                 # 仅训练第一层BN
                 partial_bn=False):
        super(ResNet3DRecognizer, self).__init__()

        self.fix_bn = fix_bn
        self.partial_bn = partial_bn

        block_layer, layer_blocks, groups, width_per_group = arch_settings[arch]
        state_dict_2d = load_state_dict_from_url(model_urls[arch], progress=True) \
            if torchvision_pretrained else None

        self.backbone = ResNet3DBackbone(
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

        feature_dims = layer_planes[-1] * block_layer.expansion
        self.head = ResNet3DHead(
            feature_dims=feature_dims,
            dropout_rate=dropout_rate,
            num_classes=pretrained_num_classes
        )

        self.init_weights(pretrained=pretrained,
                          pretrained_num_classes=pretrained_num_classes,
                          num_classes=num_classes)

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

            nn.init.normal_(self.head.fc.weight, 0, 0.01)
            nn.init.zeros_(self.head.fc.bias)

    def train(self, mode: bool = True) -> T:
        super(ResNet3DRecognizer, self).train(mode=mode)

        if mode and (self.partial_bn or self.fix_bn):
            freezing_bn(self, partial_bn=self.partial_bn)

        return self

    def forward(self, x):
        x = x.unsqueeze(2)

        x = self.backbone(x)
        x = self.head(x)

        return {KEY_OUTPUT: x}


@registry.RECOGNIZER.register('ResNet3D')
def build_resnet3d(cfg):
    # for recognizer
    arch = cfg.MODEL.BACKBONE.ARCH
    pretrained = cfg.MODEL.RECOGNIZER.PRETRAINED
    torchvision_pretrained = cfg.MODEL.RECOGNIZER.TORCHVISION_PRETRAINED
    pretrained_num_classes = cfg.MODEL.RECOGNIZER.PRETRAINED_NUM_CLASSES
    fix_bn = cfg.MODEL.NORM.FIX_BN
    partial_bn = cfg.MODEL.NORM.PARTIAL_BN
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
    down_samples = cfg.MODEL.BACKBONE.DOWN_SAMPLES
    temporal_strides = cfg.MODEL.BACKBONE.TEMPORAL_STRIDES
    inflate_list = cfg.MODEL.BACKBONE.INFLATE_LIST
    inflate_style = cfg.MODEL.BACKBONE.INFLATE_STYLE
    # for head
    dropout_rate = cfg.MODEL.HEAD.DROPOUT
    num_classes = cfg.MODEL.HEAD.NUM_CLASSES

    return ResNet3DRecognizer(
        # for backbone
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
        down_samples=down_samples,
        temporal_strides=temporal_strides,
        inflate_list=inflate_list,
        inflate_style=inflate_style,
        conv_layer=conv_layer,
        norm_layer=norm_layer,
        act_layer=act_layer,
        zero_init_residual=zero_init_residual,
        # for head
        dropout_rate=dropout_rate,
        num_classes=num_classes,
        # for recognizer
        arch=arch,
        torchvision_pretrained=torchvision_pretrained,
        pretrained=pretrained,
        pretrained_num_classes=pretrained_num_classes,
        fix_bn=fix_bn,
        partial_bn=partial_bn
    )

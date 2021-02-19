# -*- coding: utf-8 -*-

"""
@date: 2020/12/24 下午7:38
@file: shufflenetv1.py
@author: zj
@description: 
"""
from abc import ABC

import torch.nn as nn
from torch.nn.modules.module import T
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.shufflenetv2 import shufflenet_v2_x0_5, shufflenet_v2_x1_0, \
    shufflenet_v2_x1_5, shufflenet_v2_x2_0

from zcls.config.key_word import KEY_OUTPUT
from .. import registry
from zcls.model.backbones.shufflenet.shufflenetv2_unit import ShuffleNetV2Unit
from zcls.model.backbones.shufflenet.shufflenetv2_backbone import ShuffleNetV2Backbone
from ..heads.shufflenetv2_head import ShuffleNetV2Head
from ..norm_helper import get_norm, freezing_bn
from ..conv_helper import get_conv
from ..act_helper import get_act

"""
G1) Equal channel width minimizes memory access cost (MAC)
G2) Excessive group convolution increases MAC.
G3) Network fragmentation reduces degree of parallelism
G4) Element-wise operations are non-negligible.
"""

arch_settings = {
    'shufflenet_v2_x2_0': (ShuffleNetV2Unit, (244, 488, 976), (4, 8, 4), 1024),
    'shufflenet_v2_x1_5': (ShuffleNetV2Unit, (176, 352, 704), (4, 8, 4), 1024),
    'shufflenet_v2_x1_0': (ShuffleNetV2Unit, (116, 232, 464), (4, 8, 4), 1024),
    'shufflenet_v2_x0_5': (ShuffleNetV2Unit, (48, 96, 192), (4, 8, 4), 2048),
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ShuffleNetV2Recognizer(nn.Module, ABC):

    def __init__(self,
                 ############### for recognizer
                 # ZCls预训练模型
                 zcls_pretrained="",
                 # 假定预训练模型类别数
                 pretrained_num_classes=1000,
                 # 固定BN
                 fix_bn=False,
                 # 仅训练第一层BN
                 partial_bn=False,
                 # 架构
                 arch='shufflenet_v2_x2_0',
                 ########################## for head
                 # 随机失火概率
                 dropout_rate=0.,
                 # 类别数
                 num_classes=1000,
                 ############### for backbone
                 # 输入通道数
                 in_planes=3,
                 # 第一个卷积层通道数
                 base_planes=24,
                 # 是否执行空间下采样
                 down_samples=(1, 1, 1),
                 # 卷积层类型
                 conv_layer=None,
                 # 归一化层类型
                 norm_layer=None,
                 # 激活层类型
                 act_layer=None,
                 ##################### for compression
                 # 设置每一层通道数均为8的倍数
                 round_nearest=8
                 ):
        super(ShuffleNetV2Recognizer, self).__init__()
        self.fix_bn = fix_bn
        self.partial_bn = partial_bn

        block_layer, layer_planes, layer_blocks, out_planes = arch_settings[arch]

        base_planes = _make_divisible(base_planes, round_nearest)
        stage_planes = list()
        for i in range(len(layer_planes)):
            stage_planes.append(_make_divisible(layer_planes[i], round_nearest))
        layer_planes = stage_planes
        out_planes = _make_divisible(out_planes, round_nearest)

        self.backbone = ShuffleNetV2Backbone(
            # 输入通道数
            in_planes=in_planes,
            # 第一个卷积层通道数
            base_planes=base_planes,
            # 输出通道数
            out_planes=out_planes,
            # 每一层通道数
            layer_planes=layer_planes,
            # 每一层块个数
            layer_blocks=layer_blocks,
            # 是否执行空间下采样
            down_samples=down_samples,
            # 块类型
            block_layer=block_layer,
            # 卷积层类型
            conv_layer=conv_layer,
            # 归一化层类型
            norm_layer=norm_layer,
            # 激活层类型
            act_layer=act_layer
        )
        self.head = ShuffleNetV2Head(
            feature_dims=out_planes,
            dropout_rate=dropout_rate,
            num_classes=pretrained_num_classes
        )

        self.init_weights(zcls_pretrained,
                          pretrained_num_classes,
                          num_classes)

    def init_weights(self, pretrained, pretrained_num_classes, num_classes):
        if pretrained != "":
            state_dict = load_state_dict_from_url(pretrained, progress=True)
            self.load_state_dict(state_dict=state_dict, strict=False)
        if num_classes != pretrained_num_classes:
            fc = self.head.fc
            fc_features = fc.in_features
            self.head.fc = nn.Linear(fc_features, num_classes)
            self.head.init_weights()

    def train(self, mode: bool = True) -> T:
        super(ShuffleNetV2Recognizer, self).train(mode=mode)

        if mode and (self.partial_bn or self.fix_bn):
            freezing_bn(self, partial_bn=self.partial_bn)

        return self

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)

        return {KEY_OUTPUT: x}


class TorchvisionShuffleNetV2(nn.Module, ABC):

    def __init__(self,
                 arch="shufflenet_v2_x2_0",
                 num_classes=1000,
                 torchvision_pretrained=False,
                 pretrained_num_classes=1000,
                 fix_bn=False,
                 partial_bn=False):
        super(TorchvisionShuffleNetV2, self).__init__()

        self.fix_bn = fix_bn
        self.partial_bn = partial_bn

        if arch == 'shufflenet_v2_x2_0':
            self.model = shufflenet_v2_x2_0(pretrained=torchvision_pretrained, num_classes=pretrained_num_classes)
        elif arch == 'shufflenet_v2_x1_5':
            self.model = shufflenet_v2_x1_5(pretrained=torchvision_pretrained, num_classes=pretrained_num_classes)
        elif arch == 'shufflenet_v2_x1_0':
            self.model = shufflenet_v2_x1_0(pretrained=torchvision_pretrained, num_classes=pretrained_num_classes)
        elif arch == 'shufflenet_v2_x0_5':
            self.model = shufflenet_v2_x0_5(pretrained=torchvision_pretrained, num_classes=pretrained_num_classes)
        else:
            raise ValueError('no such value')

        self.init_weights(num_classes, pretrained_num_classes)

    def init_weights(self, num_classes, pretrained_num_classes):
        if num_classes != pretrained_num_classes:
            fc = self.model.fc
            fc_features = fc.in_features
            self.model.fc = nn.Linear(fc_features, num_classes)

            nn.init.normal_(self.model.fc.weight, 0, 0.01)
            nn.init.zeros_(self.model.fc.bias)

    def train(self, mode: bool = True) -> T:
        super(TorchvisionShuffleNetV2, self).train(mode=mode)

        if mode and (self.partial_bn or self.fix_bn):
            freezing_bn(self, partial_bn=self.partial_bn)

        return self

    def forward(self, x):
        x = self.model(x)

        return {KEY_OUTPUT: x}


@registry.RECOGNIZER.register('ShuffleNetV2')
def build_shufflenet_v2(cfg):
    recognizer_name = cfg.MODEL.RECOGNIZER.NAME
    # for recognizer
    torchvision_pretrained = cfg.MODEL.RECOGNIZER.TORCHVISION_PRETRAINED
    pretrained_num_classes = cfg.MODEL.RECOGNIZER.PRETRAINED_NUM_CLASSES
    # for head
    num_classes = cfg.MODEL.HEAD.NUM_CLASSES
    # for bcckbone
    arch = cfg.MODEL.BACKBONE.ARCH
    # norm
    fix_bn = cfg.MODEL.NORM.FIX_BN
    partial_bn = cfg.MODEL.NORM.PARTIAL_BN

    if recognizer_name == 'TorchvisionShuffleNetV2':
        return TorchvisionShuffleNetV2(arch=arch,
                                       num_classes=num_classes,
                                       torchvision_pretrained=torchvision_pretrained,
                                       pretrained_num_classes=pretrained_num_classes,
                                       fix_bn=fix_bn,
                                       partial_bn=partial_bn)
    elif recognizer_name == 'ShuffleNetV2Recognizer':
        # for recognizer
        pretrained = cfg.MODEL.RECOGNIZER.PRETRAINED
        # for head
        dropout_rate = cfg.MODEL.HEAD.DROPOUT
        # for bcckbone
        in_planes = cfg.MODEL.BACKBONE.IN_PLANES
        base_planes = cfg.MODEL.BACKBONE.BASE_PLANES
        down_samples = cfg.MODEL.BACKBONE.DOWN_SAMPLES
        # layer
        conv_layer = get_conv(cfg)
        norm_layer = get_norm(cfg)
        act_layer = get_act(cfg)
        # for compression
        round_nearest = cfg.MODEL.COMPRESSION.ROUND_NEAREST
        return ShuffleNetV2Recognizer(
            ############### for recognizer
            # ZCls预训练模型
            zcls_pretrained=pretrained,
            # 假定预训练模型类别数
            pretrained_num_classes=pretrained_num_classes,
            # 固定BN
            fix_bn=fix_bn,
            # 仅训练第一层BN
            partial_bn=partial_bn,
            # 架构
            arch=arch,
            ########################## for head
            # 随机失火概率
            dropout_rate=dropout_rate,
            # 类别数
            num_classes=num_classes,
            ############### for backbone
            # 输入通道数
            in_planes=in_planes,
            # 第一个卷积层通道数
            base_planes=base_planes,
            # 是否执行空间下采样
            down_samples=down_samples,
            # 卷积层类型
            conv_layer=conv_layer,
            # 归一化层类型
            norm_layer=norm_layer,
            # 激活层类型
            act_layer=act_layer,
            ##################### for compression
            # 设置每一层通道数均为8的倍数
            round_nearest=round_nearest
        )
    else:
        raise ValueError(f'{recognizer_name} does not exist')

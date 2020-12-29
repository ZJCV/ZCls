# -*- coding: utf-8 -*-

"""
@date: 2020/12/24 下午7:38
@file: shufflenetv1_recognizer.py
@author: zj
@description: 
"""
from abc import ABC

import torch.nn as nn
from torchvision.models import mnasnet
from torch.nn.modules.module import T
from torchvision.models.utils import load_state_dict_from_url

from .. import registry
from ..backbones.shufflenetv2_backbone import ShuffleNetV2Backbone
from ..heads.shufflenetv2_head import ShuffleNetV2Head
from ..norm_helper import get_norm, freezing_bn

"""
G1) Equal channel width minimizes memory access cost (MAC)
G2) Excessive group convolution increases MAC.
G3) Network fragmentation reduces degree of parallelism
G4) Element-wise operations are non-negligible.
"""

arch_settings = {
    'shufflenetv2_2x': ((244, 488, 976), 1024),
    'shufflenetv2_1x': ((176, 352, 704), 1024),
    'shufflenetv2_0.5x': ((116, 232, 464), 1024),
    'shufflenetv2_0.25x': ((48, 96, 192), 2048),
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
                 # 输入通道数
                 inplanes=3,
                 # 输出通道数
                 out_planes=1024,
                 # 类别数
                 num_classes=1000,
                 # 第一个卷积层通道数
                 base_channel=24,
                 # 每一层通道数
                 layer_planes=(176, 352, 704),
                 # 每一层块个数
                 layer_blocks=(4, 8, 4),
                 # 设置每一层通道数均为8的倍数
                 round_nearest=8,
                 # ZCls预训练模型
                 zcls_pretrained="",
                 # 假定预训练模型类别数
                 pretrained_num_classes=1000,
                 # 固定BN
                 fix_bn=False,
                 # 仅训练第一层BN
                 partial_bn=False,
                 # 卷积层类型
                 conv_layer=None,
                 # 归一化层类型
                 norm_layer=None,
                 # 激活层类型
                 act_layer=None
                 ):
        super(ShuffleNetV2Recognizer, self).__init__()
        self.num_classes = num_classes
        self.fix_bn = fix_bn
        self.partial_bn = partial_bn

        base_channel = _make_divisible(base_channel, round_nearest)
        for i in range(len(layer_planes)):
            layer_planes[i] = _make_divisible(layer_planes[i], round_nearest)
        out_planes = _make_divisible(out_planes, round_nearest)

        self.backbone = ShuffleNetV2Backbone(
            inplanes=inplanes,
            base_channel=base_channel,
            out_planes=out_planes,
            layer_planes=layer_planes,
            layer_blocks=layer_blocks,
            conv_layer=conv_layer,
            norm_layer=norm_layer,
            act_layer=act_layer
        )
        self.head = ShuffleNetV2Head(
            feature_dims=out_planes,
            num_classes=pretrained_num_classes
        )

        self.init_weights(pretrained=zcls_pretrained, pretrained_num_classes=pretrained_num_classes)

    def init_weights(self, pretrained="", pretrained_num_classes=1000):
        if pretrained != "":
            state_dict = load_state_dict_from_url(pretrained, progress=True)
            self.load_state_dict(state_dict=state_dict, strict=False)
        if self.num_classes != pretrained_num_classes:
            fc = self.head.fc
            fc_features = fc.in_features
            self.head.fc = nn.Linear(fc_features, self.num_classes)

            nn.init.normal_(self.head.fc.weight, 0, 0.01)
            nn.init.zeros_(self.head.fc.bias)

    def train(self, mode: bool = True) -> T:
        super(ShuffleNetV2Recognizer, self).train(mode=mode)

        if mode and (self.partial_bn or self.fix_bn):
            freezing_bn(self, partial_bn=self.partial_bn)

        return self

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)

        return {'probs': x}


@registry.RECOGNIZER.register('ShuffleNetV2')
def build_shufflenetv2(cfg):
    type = cfg.MODEL.RECOGNIZER.NAME
    zcls_pretrained = cfg.MODEL.ZClS_PRETRAINED
    pretrained_num_classes = cfg.MODEL.PRETRAINED_NUM_CLASSES
    arch = cfg.MODEL.BACKBONE.ARCH
    num_classes = cfg.MODEL.HEAD.NUM_CLASSES
    fix_bn = cfg.MODEL.NORM.FIX_BN
    partial_bn = cfg.MODEL.NORM.PARTIAL_BN

    norm_layer = get_norm(cfg)

    layer_planes, out_planes = arch_settings[arch]

    return ShuffleNetV2Recognizer(
        num_classes=num_classes,
        out_planes=out_planes,
        layer_planes=layer_planes,
        zcls_pretrained=zcls_pretrained,
        pretrained_num_classes=pretrained_num_classes,
        fix_bn=fix_bn,
        partial_bn=partial_bn,
        norm_layer=norm_layer
    )

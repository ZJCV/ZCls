# -*- coding: utf-8 -*-

"""
@date: 2020/11/25 下午6:49
@file: model.py
@author: zj
@description: 
"""

from yacs.config import CfgNode as CN


def add_config(_C):
    # ---------------------------------------------------------------------------- #
    # Model
    # ---------------------------------------------------------------------------- #
    _C.MODEL = CN()

    # ---------------------------------------------------------------------------- #
    # Convolution
    # ---------------------------------------------------------------------------- #
    _C.MODEL.CONV = CN()
    _C.MODEL.CONV.TYPE = 'Conv2d'

    # ---------------------------------------------------------------------------- #
    # Normalization
    # ---------------------------------------------------------------------------- #
    _C.MODEL.NORM = CN()
    _C.MODEL.NORM.TYPE = 'BatchNorm2d'
    # for BN
    _C.MODEL.NORM.SYNC_BN = False
    _C.MODEL.NORM.FIX_BN = False
    _C.MODEL.NORM.PARTIAL_BN = False
    # Precise BN stats.
    _C.MODEL.NORM.PRECISE_BN = False
    # Number of samples use to compute precise bn.
    _C.MODEL.NORM.NUM_BATCHES_PRECISE = 200
    # for GN
    _C.MODEL.NORM.GROUPS = 32

    # ---------------------------------------------------------------------------- #
    # activation
    # ---------------------------------------------------------------------------- #
    _C.MODEL.ACT = CN()
    _C.MODEL.ACT.TYPE = 'ReLU'
    _C.MODEL.ACT.SIGMOID_TYPE = 'Sigmoid'

    # ---------------------------------------------------------------------------- #
    # compression
    # ---------------------------------------------------------------------------- #
    _C.MODEL.COMPRESSION = CN()
    _C.MODEL.COMPRESSION.WIDTH_MULTIPLIER = 1.0
    # 设置每一层通道数均为8的倍数
    _C.MODEL.COMPRESSION.ROUND_NEAREST = 8

    # ---------------------------------------------------------------------------- #
    # attention
    # ---------------------------------------------------------------------------- #
    _C.MODEL.ATTENTION = CN()
    _C.MODEL.ATTENTION.WITH_ATTENTION = False
    _C.MODEL.ATTENTION.WITH_ATTENTIONS = (0, 0, 0, 0)
    _C.MODEL.ATTENTION.REDUCTION = 16
    _C.MODEL.ATTENTION.ATTENTION_TYPE = 'SqueezeAndExcitationBlock2D'

    # ---------------------------------------------------------------------------- #
    # backbone
    # ---------------------------------------------------------------------------- #
    _C.MODEL.BACKBONE = CN()
    # 输入通道数
    _C.MODEL.BACKBONE.IN_PLANES = 3
    # for ResNet series
    _C.MODEL.BACKBONE.ARCH = 'resnet18'
    # stem通道数,
    _C.MODEL.BACKBONE.BASE_PLANES = 64
    # 每一层基础通道数
    _C.MODEL.BACKBONE.LAYER_PLANES = (64, 128, 256, 512)
    # 是否执行空间下采样
    _C.MODEL.BACKBONE.DOWN_SAMPLES = (0, 1, 1, 1)
    # for ResNetSt
    # 每个group中的分离数
    _C.MODEL.BACKBONE.RADIX = 1
    # 在3x3之前执行下采样操作
    _C.MODEL.BACKBONE.FAST_AVG = False
    # for MobileNetV1
    # 每层步长
    _C.MODEL.BACKBONE.STRIDES = (1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 2)
    # for MobileNetV2
    # 输出特征维数
    _C.MODEL.BACKBONE.FEATURE_DIMS = 1280
    # for ResNet3D
    _C.MODEL.BACKBONE.CONV1_KERNEL = (1, 7, 7)
    _C.MODEL.BACKBONE.CONV1_STRIDE = (1, 2, 2)
    _C.MODEL.BACKBONE.CONV1_PADDING = (0, 3, 3)
    _C.MODEL.BACKBONE.POOL1_KERNEL = (1, 3, 3)
    _C.MODEL.BACKBONE.POOL1_STRIDE = (1, 2, 2)
    _C.MODEL.BACKBONE.POOL1_PADDING = (0, 1, 1)
    _C.MODEL.BACKBONE.WITH_POOL2 = False
    _C.MODEL.BACKBONE.TEMPORAL_STRIDES = (1, 1, 1, 1)
    _C.MODEL.BACKBONE.INFLATE_LIST = (0, 0, 0, 0)
    _C.MODEL.BACKBONE.INFLATE_STYLE = '3x1x1'
    # for ShuffleNetV1
    _C.MODEL.BACKBONE.WITH_GROUPS = (0, 1, 1)

    # ---------------------------------------------------------------------------- #
    # head
    # ---------------------------------------------------------------------------- #
    _C.MODEL.HEAD = CN()
    _C.MODEL.HEAD.DROPOUT = 0.
    _C.MODEL.HEAD.NUM_CLASSES = 1000

    # ---------------------------------------------------------------------------- #
    # recognizer
    # ---------------------------------------------------------------------------- #
    _C.MODEL.RECOGNIZER = CN()
    _C.MODEL.RECOGNIZER.TYPE = 'ResNet'
    _C.MODEL.RECOGNIZER.NAME = 'ZClsResNet'
    # zcls框架训练的模型，用于测试阶段
    _C.MODEL.RECOGNIZER.PRELOADED = ""
    # zcls框架训练的模型，用于训练阶段
    _C.MODEL.RECOGNIZER.PRETRAINED = ""
    # torchvision训练的模型，用于训练阶段
    _C.MODEL.RECOGNIZER.TORCHVISION_PRETRAINED = False
    # 预训练模型类别数
    _C.MODEL.RECOGNIZER.PRETRAINED_NUM_CLASSES = 1000
    # 零初始化残差连接
    _C.MODEL.RECOGNIZER.ZERO_INIT_RESIDUAL = False

    # ---------------------------------------------------------------------------- #
    # criterion
    # ---------------------------------------------------------------------------- #
    _C.MODEL.CRITERION = CN()
    _C.MODEL.CRITERION.NAME = 'CrossEntropyLoss'

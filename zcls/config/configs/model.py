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
    # used in AsymmetricConvolutionBlock/RepVGGBlock
    _C.MODEL.CONV.ADD_BLOCKS = None

    # ---------------------------------------------------------------------------- #
    # Normalization
    # ---------------------------------------------------------------------------- #
    _C.MODEL.NORM = CN()
    _C.MODEL.NORM.TYPE = 'BatchNorm2d'
    # batchnorm
    _C.MODEL.NORM.SYNC_BN = False
    _C.MODEL.NORM.FIX_BN = False
    _C.MODEL.NORM.PARTIAL_BN = False
    # Precise BN stats.
    _C.MODEL.NORM.PRECISE_BN = False
    # Number of samples use to compute precise bn.
    _C.MODEL.NORM.NUM_BATCHES_PRECISE = 200
    # groupnorm
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
    # Set the number of channels in each layer to be a multiple of 8
    _C.MODEL.COMPRESSION.ROUND_NEAREST = 8

    # ---------------------------------------------------------------------------- #
    # attention
    # ---------------------------------------------------------------------------- #
    _C.MODEL.ATTENTION = CN()
    _C.MODEL.ATTENTION.WITH_ATTENTION = False
    _C.MODEL.ATTENTION.WITH_ATTENTIONS = (0, 0, 0, 0)
    _C.MODEL.ATTENTION.REDUCTION = 16
    _C.MODEL.ATTENTION.ATTENTION_TYPE = 'SqueezeAndExcitationBlock2D'
    # used for se-block
    _C.MODEL.ATTENTION.BIAS = False

    # ---------------------------------------------------------------------------- #
    # backbone
    # ---------------------------------------------------------------------------- #
    _C.MODEL.BACKBONE = CN()
    _C.MODEL.BACKBONE.NAME = 'ShuffleNetV1'
    _C.MODEL.BACKBONE.IN_PLANES = 3
    _C.MODEL.BACKBONE.ARCH = 'resnet18'
    _C.MODEL.BACKBONE.BASE_PLANES = 64
    # base channels each layer
    _C.MODEL.BACKBONE.LAYER_PLANES = (64, 128, 256, 512)
    _C.MODEL.BACKBONE.DOWNSAMPLES = (0, 1, 1, 1)
    # Is AvgPool used instead of Conv2d for downsampling in residual block
    _C.MODEL.BACKBONE.USE_AVG = False
    # Perform downsampling before 3x3
    _C.MODEL.BACKBONE.FAST_AVG = False
    # for MobileNetV1
    _C.MODEL.BACKBONE.STRIDES = (1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 2)
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
    _C.MODEL.HEAD.NAME = 'GeneralHead2D'
    _C.MODEL.HEAD.FEATURE_DIMS = 1024
    _C.MODEL.HEAD.DROPOUT_RATE = 0.
    _C.MODEL.HEAD.NUM_CLASSES = 1000
    _C.MODEL.HEAD.BIAS = True
    # for mobilenetv3
    _C.MODEL.HEAD.INNER_DIMS = 1280

    # ---------------------------------------------------------------------------- #
    # recognizer
    # ---------------------------------------------------------------------------- #
    _C.MODEL.RECOGNIZER = CN()
    _C.MODEL.RECOGNIZER.NAME = 'ShuffleNetV1'
    _C.MODEL.RECOGNIZER.TYPE = 'ResNet'
    # zcls training model, used in the test phase
    _C.MODEL.RECOGNIZER.PRELOADED = ""
    # zcls training model, used in the training stage
    _C.MODEL.RECOGNIZER.PRETRAINED = ""
    # torchvision training model, used in the training phase
    _C.MODEL.RECOGNIZER.TORCHVISION_PRETRAINED = False
    # Number of pre-training model categories
    _C.MODEL.RECOGNIZER.PRETRAINED_NUM_CLASSES = 1000
    # Zero initialization residual connection
    _C.MODEL.RECOGNIZER.ZERO_INIT_RESIDUAL = False

    # ---------------------------------------------------------------------------- #
    # criterion
    # ---------------------------------------------------------------------------- #
    _C.MODEL.CRITERION = CN()
    _C.MODEL.CRITERION.NAME = 'CrossEntropyLoss'
    # for label smoothing loss
    _C.MODEL.CRITERION.SMOOTHING = 0.1
    # mean or sum
    _C.MODEL.CRITERION.REDUCTION = 'mean'

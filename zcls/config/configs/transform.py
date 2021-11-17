# -*- coding: utf-8 -*-

"""
@date: 2020/11/25 下午6:51
@file: transform.py
@author: zj
@description: 
"""

from yacs.config import CfgNode as CN


def add_config(_C):
    # ---------------------------------------------------------------------------- #
    # Transform
    # ---------------------------------------------------------------------------- #
    _C.TRANSFORM = CN()
    _C.TRANSFORM.TRAIN_METHODS = ('Resize', 'CenterCrop', 'Normalize', 'ToTensor')
    _C.TRANSFORM.TEST_METHODS = ('Resize', 'CenterCrop', 'Normalize', 'ToTensor')

    # default: policy: AutoAugmentPolicy = AutoAugmentPolicy.IMAGENET, p=0.5
    # for policy, should be
    #     IMAGENET = "imagenet"
    #     CIFAR10 = "cifar10"
    #     SVHN = "svhn"
    _C.TRANSFORM.AUTOAUGMENT = ("imagenet", 0.5)

    # default: max_holes=8, max_height=8, max_width=8, min_holes=None, min_height=None, min_width=None, fill_value=0, p=0.5
    _C.TRANSFORM.COARSE_DROPOUT = (8, 8, 8, None, None, None, 0, 0.5)

    # default: brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5
    _C.TRANSFORM.COLOR_JITTER = (0.2, 0.2, 0.2, 0.2, 0.5)

    # default: size, p=1.0
    _C.TRANSFORM.TRAIN_CENTER_CROP = ((224, 224), 1.0)
    _C.TRANSFORM.TEST_CENTER_CROP = ((224, 224), 1.0)

    # default: size, p=1.0
    _C.TRANSFORM.RANDOM_CROP = ((224, 224), 1.0)

    # default: p=0.5
    _C.TRANSFORM.HORIZONTAL_FLIP = 0.5

    # default: p=0.5
    _C.TRANSFORM.VERTICAL_FLIP = 0.5

    # default: size, interpolation=cv2.INTER_LINEAR, mode=0, p=1.0
    # for interpolation, should be
    #     INTER_AREA = 3
    #     INTER_CUBIC = 2
    #     INTER_LANCZOS4 = 4
    #     INTER_LINEAR = 1
    #     INTER_NEAREST = 0
    # for mode, should be
    #     mode = 0 (zoom to largest edge)
    #     mode = 1 (zoom to smallest edge)
    _C.TRANSFORM.TRAIN_RESIZE = ((224, 224), 1, 0, 1.0)
    _C.TRANSFORM.TRAIN_RESIZE2 = ((224,), 1, 0, 1.0)
    _C.TRANSFORM.TEST_RESIZE = ((224, 224), 1, 0, 1.0)
    _C.TRANSFORM.TEST_RESIZE2 = ((224,), 1, 0, 1.0)

    # default: limit, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, value=None, p=0.5
    # for interpolation, should be
    #     INTER_AREA = 3
    #     INTER_CUBIC = 2
    #     INTER_LANCZOS4 = 4
    #     INTER_LINEAR = 1
    #     INTER_NEAREST = 0
    # for padding_mode, should be
    #     BORDER_CONSTANT = 0
    #     BORDER_DEFAULT = 4
    #     BORDER_ISOLATED = 16
    #     BORDER_REFLECT = 2
    #     BORDER_REFLECT101 = 4
    _C.TRANSFORM.ROTATE = ((-30, 30), 1, 4, None, 0.5)

    # default: padding_position=A.PadIfNeeded.PositionType.CENTER, padding_mode=cv2.BORDER_CONSTANT, fill=0, p=1.0
    # for padding_position, should be
    #     CENTER = "center"
    #     TOP_LEFT = "top_left"
    #     TOP_RIGHT = "top_right"
    #     BOTTOM_LEFT = "bottom_left"
    #     BOTTOM_RIGHT = "bottom_right"
    # for padding_mode, should be
    #     BORDER_CONSTANT = 0
    #     BORDER_DEFAULT = 4
    #     BORDER_ISOLATED = 16
    #     BORDER_REFLECT = 2
    #     BORDER_REFLECT101 = 4
    _C.TRANSFORM.SQUARE_PAD = ("center", 0, 0, 1.0)

    # default: mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0
    # Normalization is applied by the formula: `img = (img - mean * max_pixel_value) / (std * max_pixel_value)`
    _C.TRANSFORM.NORMALIZE = ((0.445, 0.445, 0.445), (0.225, 0.225, 0.225), 255.0, 1.0)

    # default: p=1.0
    # Convert image to `torch.Tensor`. The numpy `HWC` image is converted to pytorch `CHW` tensor.
    _C.TRANSFORM.TO_TENSOR = 1.0

# -*- coding: utf-8 -*-

"""
@date: 2020/8/22 下午9:40
@file: trainer.py
@author: zj
@description: 
"""

import torch

import torchvision.transforms.transforms as transforms
import torchvision.transforms.autoaugment as autoaugment

from .square_pad import SquarePad
from .resize import Resize

"""
current supported transforms methods:

1. 'ToTensor'
2. 'ConvertImageDtype',
3. 'Normalize',
# 4. 'Resize'
5. 'CenterCrop'
6. 'RandomCrop'
7. 'RandomHorizontalFlip'
8. 'RandomVerticalFlip'
9. 'ColorJitter'
10. 'Grayscale'
11. 'RandomRotation'
12. 'RandomErasing'
13. 'RandomAutocontrast'
14. 'RandomAdjustSharpness'
15. 'RandomPosterize'
16. 'ToPILImage'

custom methods:

1. 'SquarePad'
2. 'Resize'
"""


def parse_transform(cfg, is_train=True):
    methods = cfg.TRANSFORM.TRAIN_METHODS if is_train else cfg.TRANSFORM.TEST_METHODS
    assert isinstance(methods, tuple)

    keys = transforms.__all__
    transforms_dict = transforms.__dict__
    aug_list = list()
    for method in methods:
        if method == 'SquarePad':
            transform = SquarePad
        elif method == 'Resize':
            transform = Resize
        elif method in keys:
            transform = transforms_dict[method]
        elif method == 'AUTO_AUGMENT':
            transform = autoaugment.AutoAugment
        else:
            raise ValueError(f'f{method} does not exists')

        if method == 'SquarePad':
            aug_list.append(transform())
        elif method == 'RandomAutocontrast':
            aug_list.append(transform())
        elif method == 'RandomAdjustSharpness':
            sharpness_factor = cfg.TRANSFORM.SHARPNESS_FACTOR
            aug_list.append(transform(sharpness_factor))
        elif method == 'RandomPosterize':
            bits = cfg.TRANSFORM.BITS
            aug_list.append(transform(bits))
        elif method == 'RandomRotation':
            degree = cfg.TRANSFORM.ROTATE_DEGREE
            degree = degree[0] if len(degree) == 1 else degree
            expand = cfg.TRANSFORM.ROTATE_EXPAND
            aug_list.append(transform(degree, expand=expand))
        elif method == 'Resize':
            size = cfg.TRANSFORM.TRAIN_RESIZE if is_train else cfg.TRANSFORM.TEST_RESIZE
            aug_list.append(transform(size))
        elif method == 'CenterCrop':
            size = cfg.TRANSFORM.TRAIN_CROP if is_train else cfg.TRANSFORM.TEST_CROP
            aug_list.append(transform(size))
        elif method == 'RandomCrop':
            size = cfg.TRANSFORM.TRAIN_CROP if is_train else cfg.TRANSFORM.TEST_CROP
            aug_list.append(transform(size))
        elif method == 'RandomHorizontalFlip':
            aug_list.append(transform())
        elif method == 'RandomVerticalFlip':
            aug_list.append(transform())
        elif method == 'ColorJitter':
            brightness, contrast, saturation, hue = cfg.TRANSFORM.ColorJitter
            aug_list.append(transform(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue))
        elif method == 'AUTO_AUGMENT':
            if cfg.TRANSFORM.AUGMENT_POLICY == 'imagenet':
                policy = autoaugment.AutoAugmentPolicy.IMAGENET
            elif cfg.TRANSFORM.AUGMENT_POLICY == 'cifar10':
                policy = autoaugment.AutoAugmentPolicy.CIFAR10
            else:
                raise ValueError(f'{cfg.TRANSFORM.AUGMENT_POLICY} does not exists')
            aug_list.append(transform(policy=policy))
        elif method == 'Grayscale':
            aug_list.append(transform())
        elif method == 'ToPILImage':
            aug_list.append(transform())
        elif method == 'ToTensor':
            aug_list.append(transform())
        elif method == 'ConvertImageDtype':
            if cfg.TRANSFORM.IMAGE_DTYPE == 'uint8':
                dtype = torch.uint8
            elif cfg.TRANSFORM.IMAGE_DTYPE == 'float32':
                dtype = torch.float32
            else:
                raise ValueError(f'{cfg.TRANSFORM.IMAGE_DTYPE} does not exists')
            aug_list.append(transform(dtype))
        elif method == 'RandomErasing':
            p = cfg.TRANSFORM.ERASE_P
            scale = cfg.TRANSFORM.ERASE_SCALE
            ratio = cfg.TRANSFORM.ERASE_RATIO
            aug_list.append(transform(p=p, scale=scale, ratio=ratio))
        elif method == 'Normalize':
            aug_list.append(transform(cfg.TRANSFORM.MEAN, cfg.TRANSFORM.STD))
        else:
            raise ValueError(f'{method} does not exists')

    return transforms.Compose(aug_list)


def parse_target_transform():
    return None


def build_transform(cfg, is_train=True):
    return parse_transform(cfg, is_train), parse_target_transform()

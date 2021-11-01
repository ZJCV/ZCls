# -*- coding: utf-8 -*-

"""
@date: 2020/8/22 下午9:40
@file: trainer.py
@author: zj
@description: 
"""

import torchvision.transforms.transforms as transforms
from torchvision.transforms.autoaugment import AutoAugmentPolicy

from . import realization


def parse_transform(cfg, is_train=True):
    methods = cfg.TRANSFORM.TRAIN_METHODS if is_train else cfg.TRANSFORM.TEST_METHODS
    assert isinstance(methods, tuple)

    keys = realization.__all__
    transforms_dict = realization.__dict__
    aug_list = list()
    for method in methods:
        if method in keys:
            transform = transforms_dict[method]
        else:
            raise ValueError(f'f{method} does not exists')

        if method == 'AutoAugment':
            policy, p = cfg.TRANSFORM.AUTOAUGMENT
            if policy == 'imagenet':
                policy = AutoAugmentPolicy.IMAGENET
            elif policy == 'cifar10':
                policy = AutoAugmentPolicy.CIFAR10
            elif policy == 'svhn':
                policy = AutoAugmentPolicy.SVHN
            else:
                raise ValueError(f'{policy} does not supports')
            aug_list.append(transform(policy=policy, p=p))
        elif method == 'CoarseDropout':
            max_holes, max_height, max_width, min_holes, min_height, min_width, fill_value, p \
                = cfg.TRANSFORM.COARSE_DROPOUT
            aug_list.append(transform(max_holes=max_holes, max_height=max_height, max_width=max_width,
                                      min_holes=min_holes, min_height=min_height, min_width=min_width,
                                      fill_value=fill_value, p=p))
        elif method == 'CenterCrop':
            if is_train:
                size, p = cfg.TRANSFORM.TRAIN_CENTER_CROP
            else:
                size, p = cfg.TRANSFORM.TEST_CENTER_CROP
            aug_list.append(transform(size, p=p))
        elif method == 'ColorJitter':
            brightness, contrast, saturation, hue, p = cfg.TRANSFORM.COLOR_JITTER
            aug_list.append(transform(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue, p=p))
        elif method == 'RandomCrop':
            size, p = cfg.TRANSFORM.RANDOM_CROP
            aug_list.append(transform(size, p=p))
        elif method == 'HorizontalFlip':
            p = cfg.TRANSFORM.HORIZONTAL_FLIP
            aug_list.append(transform(p=p))
        elif method == 'VerticalFlip':
            p = cfg.TRANSFORM.VERTICAL_FLIP
            aug_list.append(transform(p=p))
        elif method == 'Resize' or method == 'Resize2':
            if is_train:
                size, interpolation, mode, p = cfg.TRANSFORM.TRAIN_RESIZE
            else:
                size, interpolation, mode, p = cfg.TRANSFORM.TEST_RESIZE
            aug_list.append(transform(size, interpolation=interpolation, mode=mode, p=p))
        elif method == 'Rotate':
            limit, interpolation, border_mode, value, p = cfg.TRANSFORM.ROTATE
            aug_list.append(transform(limit, interpolation=interpolation,
                                      border_mode=border_mode, value=value, p=p))
        elif method == 'SquarePad':
            padding_position, padding_mode, fill, p = cfg.TRANSFORM.SQUARE_PAD
            aug_list.append(transform(padding_position=padding_position, padding_mode=padding_mode, fill=fill, p=p))
        elif method == 'Normalize':
            mean, std, max_pixel_value, p = cfg.TRANSFORM.NORMALIZE
            aug_list.append(transform(mean=mean, std=std, max_pixel_value=max_pixel_value, p=p))
        elif method == 'ToTensor':
            p = cfg.TRANSFORM.TO_TENSOR
            aug_list.append(transform(p=p))
        else:
            raise ValueError(f'{method} does not exists')

    return transforms.Compose(aug_list)


def parse_target_transform():
    return None


def build_transform(cfg, is_train=True):
    return parse_transform(cfg, is_train), parse_target_transform()

# -*- coding: utf-8 -*-

"""
@date: 2020/8/22 下午9:40
@file: trainer.py
@author: zj
@description: 
"""

import torchvision.transforms as transforms


def build_transform(cfg, is_train=True):
    MEAN = cfg.TRANSFORM.MEAN
    STD = cfg.TRANSFORM.STD

    aug_list = list()
    if is_train:
        shorter_side = cfg.TRANSFORM.TRAIN.SHORTER_SIDE
        assert shorter_side > 0
        aug_list.append(transforms.Resize(shorter_side))
        if cfg.TRANSFORM.TRAIN.CENTER_CROP:
            crop_size = cfg.TRANSFORM.TRAIN.TRAIN_CROP_SIZE
            aug_list.append(transforms.CenterCrop(crop_size))

        aug_list.append(transforms.ToTensor())
        aug_list.append(transforms.Normalize(MEAN, STD))
    else:
        shorter_side = cfg.TRANSFORM.TEST.SHORTER_SIDE
        assert shorter_side > 0
        aug_list.append(transforms.Resize(shorter_side))
        if cfg.TRANSFORM.TEST.CENTER_CROP:
            crop_size = cfg.TRANSFORM.TEST.TEST_CROP_SIZE
            aug_list.append(transforms.CenterCrop(crop_size))

        aug_list.append(transforms.ToTensor())
        aug_list.append(transforms.Normalize(MEAN, STD))

    transform = transforms.Compose(aug_list)
    return transform

# -*- coding: utf-8 -*-

"""
@date: 2020/8/22 下午9:40
@file: trainer.py
@author: zj
@description: 
"""

import ztransforms.cls.transforms as transforms
import ztransforms.cls.autoaugment as autoaugment


def parse_train_transform(cfg):
    aug_list = list()

    # aug_list.append(transforms.ToPILImage())

    if cfg.TRANSFORM.TRAIN.RESIZE:
        shorter_side = cfg.TRANSFORM.TRAIN.SHORTER_SIDE
        assert shorter_side > 0
        aug_list.append(transforms.Resize(shorter_side))

    if cfg.TRANSFORM.TRAIN.CENTER_CROP:
        crop_size = cfg.TRANSFORM.TRAIN.TRAIN_CROP_SIZE
        aug_list.append(transforms.CenterCrop(crop_size))

    if cfg.TRANSFORM.TRAIN.RANDOM_CROP:
        crop_size = cfg.TRANSFORM.TRAIN.TRAIN_CROP_SIZE
        aug_list.append(transforms.RandomCrop(crop_size))

    if cfg.TRANSFORM.TRAIN.RANDOM_HORIZONTAL_FLIP:
        aug_list.append(transforms.RandomHorizontalFlip())

    if cfg.TRANSFORM.TRAIN.WITH_COLOR_JITTING:
        brightness, contrast, saturation, hue = cfg.TRANSFORM.TRAIN.COLOR_JITTING
        aug_list.append(
            transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue))

    aug_list.append(transforms.ToTensor())

    if cfg.TRANSFORM.TRAIN.AUTO_AUGMENT:
        policy = cfg.TRANSFORM.TRAIN.AUGMENT_POLICY
        aug_list.append(autoaugment.AutoAugment(policy=policy))

    aug_list.append(transforms.Normalize(cfg.TRANSFORM.MEAN, cfg.TRANSFORM.STD))

    return aug_list


def parse_test_transform(cfg):
    aug_list = list()

    # aug_list.append(transforms.ToPILImage())

    if cfg.TRANSFORM.TEST.RESIZE:
        shorter_side = cfg.TRANSFORM.TEST.SHORTER_SIDE
        assert shorter_side > 0
        aug_list.append(transforms.Resize(shorter_side))

    if cfg.TRANSFORM.TEST.CENTER_CROP:
        crop_size = cfg.TRANSFORM.TEST.TEST_CROP_SIZE
        aug_list.append(transforms.CenterCrop(crop_size))

    aug_list.append(transforms.ToTensor())

    aug_list.append(transforms.Normalize(cfg.TRANSFORM.MEAN, cfg.TRANSFORM.STD))

    return aug_list


def build_transform(cfg, is_train=True):
    if is_train:
        aug_list = parse_train_transform(cfg)
    else:
        aug_list = parse_test_transform(cfg)
    transform = transforms.Compose(aug_list)
    return transform

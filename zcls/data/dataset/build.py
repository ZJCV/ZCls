# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午4:46
@file: build.py
@author: zj
@description: 
"""

import os

import torchvision.datasets as datasets

from ..transform.build import build_transform


def build_dataset(args):
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    # valdir = os.path.join(args.data, 'val')
    valdir = os.path.join(args.data, 'test')

    train_transform, val_transform = build_transform(args)

    train_dataset = datasets.ImageFolder(
        traindir, transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        valdir, transform=val_transform
    )

    return train_dataset, val_dataset

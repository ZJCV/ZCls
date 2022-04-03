# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午4:50
@file: build.py
@author: zj
@description: 
"""

import torchvision.transforms as transforms


def build_transform(args):
    if (args.arch == "inception_v3"):
        raise RuntimeError("Currently, inception_v3 is not supported by this example.")
        # crop_size = 299
        # val_size = 320 # I chose this value arbitrarily, we can adjust.
    else:
        crop_size = 224
        val_size = 256

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(), Too slow
        # normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(val_size),
        transforms.CenterCrop(crop_size),
    ])

    return train_transform, val_transform

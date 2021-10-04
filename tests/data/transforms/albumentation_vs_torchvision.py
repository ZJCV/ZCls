# -*- coding: utf-8 -*-

"""
@date: 2021/10/4 上午10:24
@file: albumentation_vs_torchvision.py
@author: zj
@description: 
"""

import cv2
import time

import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.transforms as transforms
from tqdm import tqdm

from zcls.config import cfg
from zcls.data.transforms.build import build_transform


def get_torchvision_transform():
    return transforms.Compose([
        transforms.RandomRotation((-30, 30)),
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ColorJitter(brightness=0.2, contrast=0., saturation=0., hue=0.),
        transforms.ToTensor(),
        transforms.RandomErasing(),
        transforms.Normalize([0.445, 0.445, 0.445], [0.225, 0.225, 0.225]),
    ])


def get_albumentation_transform():
    # Declare an augmentation pipeline
    return A.Compose([
        A.Rotate(limit=(-30, 30), interpolation=cv2.INTER_LINEAR, p=1.0),
        A.Resize(256, 256, interpolation=cv2.INTER_LINEAR, p=1.0),
        A.RandomCrop(width=224, height=224, p=1.0),
        A.HorizontalFlip(p=1.0),
        A.ColorJitter(brightness=0.2, contrast=0., saturation=0., p=1.0),
        A.CoarseDropout(max_height=16, max_width=16, fill_value=0, p=1.0),
        A.Normalize(mean=[0.445, 0.445, 0.445], std=[0.225, 0.225, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ])


def compute_time(transform, is_pil=False, is_a=False):
    t = 0.
    for i in tqdm(range(1000)):
        data = np.random.randn(224, 224, 3).astype(np.uint8)
        if is_pil:
            data = Image.fromarray(data)

        t1 = time.time()
        if is_pil:
            transform(data)
        else:
            if is_a:
                transform(image=data)
            else:
                transform(data)
        t += time.time() - t1

    print('average time is:', (t / 1000))


if __name__ == '__main__':
    t = get_torchvision_transform()
    print('torchvision:', t)
    compute_time(t, is_pil=True)

    t = get_albumentation_transform()
    print('albumentation:', t)
    compute_time(t, is_pil=False, is_a=True)

    cfg_file = 'tests/configs/transforms_total.yaml'
    cfg.merge_from_file(cfg_file)

    t_train, _ = build_transform(cfg, is_train=True)
    print('zcls:', t_train)
    compute_time(t_train, is_pil=False, is_a=False)

# -*- coding: utf-8 -*-

"""
@date: 2021/9/24 下午5:18
@file: get_cifar_data.py
@author: zj
@description: 
"""

import os

from PIL import Image
from tqdm import tqdm

from zcls.config import cfg
from zcls.data.datasets.build import build_dataset


def get_dataset(cfg_file, is_train=False):
    cfg.merge_from_file(cfg_file)
    dataset = build_dataset(cfg, is_train=is_train)

    return dataset


def process(dataset, dst_root):
    assert not os.path.exists(dst_root)
    os.makedirs(dst_root)

    classes = dataset.classes
    for i, (image, target) in tqdm(enumerate(iter(dataset))):
        assert isinstance(image, Image.Image)

        cls_name = classes[target]
        cls_dir = os.path.join(dst_root, cls_name)
        if not os.path.exists(cls_dir):
            os.makedirs(cls_dir)

        img_path = os.path.join(cls_dir, f'{i}.jpg')
        assert not os.path.exists(img_path)
        image.save(img_path)


if __name__ == '__main__':
    cfg_file = 'tools/data/cifar100.yaml'
    test_dataset = get_dataset(cfg_file, is_train=False)

    dst_root = '/home/zj/data/cifar/test'
    process(test_dataset, dst_root)

    train_dataset = get_dataset(cfg_file, is_train=True)
    dst_root = '/home/zj/data/cifar/train'
    process(train_dataset, dst_root)

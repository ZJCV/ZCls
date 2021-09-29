# -*- coding: utf-8 -*-

"""
@date: 2021/9/24 下午5:18
@file: get_cifar_data.py
@author: zj
@description: 
"""

import os
import json

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


def process_v2(dataset, dst_root):
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)
    assert os.path.isdir(dst_root)

    img_list = list()
    target_list = list()
    class_list = dataset.classes
    for i, (image, target) in tqdm(enumerate(iter(dataset))):
        assert isinstance(image, Image.Image)

        cls_name = class_list[target]
        cls_dir = os.path.join(dst_root, cls_name)
        if not os.path.exists(cls_dir):
            os.makedirs(cls_dir)

        img_path = os.path.join(cls_dir, f'{i}.jpg')
        if not os.path.exists(img_path):
            image.save(img_path)

        img_list.append(img_path)
        target_list.append(target)

    return {
        'imgs': img_list,
        'classes': class_list,
        'targets': target_list
    }


def save_to_json(data_dict, json_path):
    assert isinstance(data_dict, dict)
    assert not os.path.exists(json_path)

    with open(json_path, 'w') as f:
        json.dump(data_dict, f)


if __name__ == '__main__':
    print('test ...')
    cfg_file = 'tools/data/cifar100.yaml'
    test_dataset = get_dataset(cfg_file, is_train=False)

    dst_root = '/home/zj/data/cifar/test'
    data_dict = process_v2(test_dataset, dst_root)

    json_path = '/home/zj/data/cifar/cifar100_test.json'
    save_to_json(data_dict, json_path)

    print('train ...')
    train_dataset = get_dataset(cfg_file, is_train=True)

    dst_root = '/home/zj/data/cifar/train'
    data_dict = process_v2(train_dataset, dst_root)

    json_path = '/home/zj/data/cifar/cifar100_train.json'
    save_to_json(data_dict, json_path)

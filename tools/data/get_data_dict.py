# -*- coding: utf-8 -*-

"""
@date: 2021/9/24 下午5:49
@file: get_data_dict.py
@author: zj
@description: 
"""
import glob
import json
import os


def get_data(data_root):
    assert os.path.isdir(data_root)

    data_dict = dict()

    class_list = os.listdir(data_root)
    for cls_name in class_list:
        cls_dir = os.path.join(data_root, cls_name)
        img_list = glob.glob(os.path.join(cls_dir, '*.jpg'))

        data_dict[cls_name] = img_list

    return data_dict


def save_to_json(data_dict, dst_path):
    assert isinstance(data_dict, dict)
    assert not os.path.isfile(dst_path)

    with open(dst_path, 'w') as f:
        json.dump(data_dict, f)


if __name__ == '__main__':
    print('process train')
    train_root = '/home/zj/data/cifar/train'
    data_dict = get_data(train_root)

    dst_path = '/home/zj/data/cifar/cifar100_train.json'
    save_to_json(data_dict, dst_path)

    print('process test')
    test_root = '/home/zj/data/cifar/test'
    data_dict = get_data(test_root)

    dst_path = '/home/zj/data/cifar/cifar100_test.json'
    save_to_json(data_dict, dst_path)

    print('done')

# -*- coding: utf-8 -*-

"""
@date: 2021/4/7 下午7:10
@file: zoom.py
@author: zj
@description: Assume that the image source directory is arranged as follows:
root/dog/xxx.png
root/dog/xxy.png
root/dog/xxz.png

root/cat/123.png
root/cat/nsdf3.png
root/cat/asd932_.png
"""

import os
from ztransforms.cls import Resize
from PIL import Image
from multiprocessing import Pool

num_worker = int(os.cpu_count() / 2)
# Specifies the size of the shorter edge of the image
short_side = 100
# After processing, the image is saved in PNG format
suffix = '.png'

# Dataset source directory
src_dir = 'data/imagenet/train'
# src_dir = 'data/imagenet/val'
# Dataset results directory
dst_dir = 'data/imagenet/train_zoom'


# dst_dir = 'data/imagenet/val_zoom'


def batch_process(cate_name):
    # Resize
    model = get_model(short_side)
    # for i, cate_name in enumerate(cate_list):
    src_cate_dir = os.path.join(src_dir, cate_name)
    dst_cate_dir = os.path.join(dst_dir, cate_name)
    if not os.path.exists(dst_cate_dir):
        os.makedirs(dst_cate_dir)

    print(f'begin {src_cate_dir}')
    file_list = os.listdir(src_cate_dir)
    for file_name in file_list:
        img_name, src_suffix = os.path.splitext(file_name)
        src_file_path = os.path.join(src_cate_dir, file_name)
        dst_file_path = os.path.join(dst_cate_dir, img_name + suffix)
        if os.path.isfile(dst_file_path):
            continue

        # fix OSError: cannot write mode RGBA as JPEG
        src_img = Image.open(src_file_path).convert('RGB')
        dst_img = model(src_img)

        dst_img.save(dst_file_path)
    print(f'end {src_cate_dir}')


def get_model(short_side):
    model = Resize(short_side)
    return model


if __name__ == '__main__':
    if not os.path.exists(src_dir):
        raise ValueError(f'{src_dir} does not exists')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    cate_list = os.listdir(src_dir)

    pool = Pool(num_worker)
    pool.map(batch_process, cate_list)

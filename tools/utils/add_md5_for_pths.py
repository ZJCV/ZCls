# -*- coding: utf-8 -*-

"""
@date: 2021/11/14 上午10:58
@file: add_md5_for_pths.py
@author: zj
@description: Add md5 check code to each pth file (only the last 8 digits)
Usage:
1. add md5 for one file:
    ```
    $ python add_md5_for_pths.py weight_file_path dst_root --verbose
    ```
2. add md5 for batch files:
    ```
    $ python add_md5_for_pths.py weight_file_root dst_root --batch --verbose
    ```
"""

import os
import hashlib
import shutil
import argparse

from tqdm import tqdm
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Add md5 for weight file')
    parser.add_argument("src",
                        type=str,
                        default="",
                        help="Path to weight file")
    parser.add_argument('dst',
                        type=str,
                        default="",
                        help="Path to output")
    parser.add_argument('--batch',
                        default=False,
                        action='store_true',
                        help="Batch processing, when set True, enter src path as directory.")
    parser.add_argument('--verbose',
                        default=False,
                        action='store_true',
                        help="Print Info")

    args = parser.parse_args()
    return args


def get_files(data_root):
    assert os.path.isdir(data_root)

    p = Path(data_root)
    files = p.rglob('*.pth')
    files_list = [x for x in files]

    return files_list


def get_md5(file_path):
    assert os.path.isfile(file_path)

    hs = hashlib.md5()
    with open(file_path, mode='rb') as f:
        while True:
            content = f.read(3)
            if content:
                hs.update(content)
            else:
                break
        return hs.hexdigest()


def process(data_list, dst_root):
    for file_path in tqdm(data_list):
        md5_code = get_md5(file_path)

        file_dir, file_name = os.path.split(file_path)[:2]
        name_prefix, name_suffix = os.path.splitext(file_name)[:2]

        dst_file_path = os.path.join(dst_root, f'{name_prefix}_{str(md5_code[-8:])}{name_suffix}')
        assert not os.path.exists(dst_file_path)

        shutil.copy(file_path, dst_file_path)


if __name__ == '__main__':
    args = parse_args()

    src_path = args.src
    dst_path = os.path.abspath(args.dst)
    is_batch = args.batch
    is_verbose = args.verbose

    if is_batch:
        assert os.path.isdir(src_path), f'{src_path} is not a dir'
        if is_verbose:
            print(f'[SEARCH]: get files from {src_path}')
        files_list = get_files(src_path)
    else:
        assert os.path.isfile(src_path), f'{src_path} is not a file'
        if is_verbose:
            print(f'[SEARCH]: process file {src_path}')
        files_list = [src_path, ]

    if is_verbose:
        print('[PROCESS]: add md5')
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    assert os.path.isdir(dst_path), f'{dst_path} is not a dir'
    process(files_list, dst_path)

    print('done')

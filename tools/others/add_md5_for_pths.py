# -*- coding: utf-8 -*-

"""
@date: 2021/11/14 上午10:58
@file: add_md5_for_pths.py
@author: zj
@description: 对每个pth文件添加md5校验码（仅取后8位）
"""

import os
import hashlib
import shutil

from tqdm import tqdm
from pathlib import Path


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


def process(data_list):
    assert isinstance(data_list, list)

    for file_path in tqdm(data_list):
        md5_code = get_md5(file_path)

        file_dir, file_name = os.path.split(file_path)[:2]
        name_prefix, name_suffix = os.path.splitext(file_name)[:2]

        dst_file_path = os.path.join(file_dir, f'{name_prefix}_{str(md5_code[-8:])}{name_suffix}')
        assert not os.path.exists(dst_file_path)

        shutil.copy(file_path, dst_file_path)


if __name__ == '__main__':
    data_root = '/home/zj/repos/ZCls/outputs/converters'

    print('get files ...')
    files_list = get_files(data_root)

    print('process ...')
    process(files_list)

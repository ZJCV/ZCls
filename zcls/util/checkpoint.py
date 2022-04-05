# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午1:40
@file: checkpoint.py
@author: zj
@description: 
"""

import os
import torch
import shutil


def save_checkpoint(state, is_best, output_dir='outputs', filename='checkpoint.pth.tar'):
    save_path = os.path.join(output_dir, filename)
    torch.save(state, save_path)
    if is_best:
        shutil.copyfile(save_path, os.path.join(output_dir, 'model_best.pth.tar'))
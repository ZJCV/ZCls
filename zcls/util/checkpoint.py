# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午1:40
@file: checkpoint.py
@author: zj
@description: 
"""

import torch
import shutil


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

# -*- coding: utf-8 -*-

"""
@date: 2020/10/22 上午9:30
@file: util.py
@author: zj
@description: 
"""

import cv2
import numpy as np
import torch


def process_cv2_inputs(frames, cfg, transform):
    """
    Normalize and prepare inputs as a list of tensors. Each tensor
    correspond to a unique pathway.
    Args:
        frames (list of array): list of input images (correspond to one clip) in range [0, 255].
        cfg (CfgNode): configs. Details can be found in
            pycls/config/defaults.py
    """
    num_clips = cfg.DATASETS.NUM_CLIPS
    index = np.linspace(0, len(frames) - 1, num=num_clips).astype(np.int)
    if cfg.VISUALIZATION.INPUT_FORMAT == "BGR":
        frames = [
            cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB) for i in index
        ]

    image_list = [transform(frame) for frame in frames]
    # [T, C, H, W] -> [C, T, H, W]
    image = torch.stack(image_list).transpose(0, 1)
    # [C, T, H, W] -> [1, C, T, H, W]
    inputs = image.unsqueeze(0)
    return inputs

# -*- coding: utf-8 -*-

"""
@date: 2021/10/4 下午1:26
@file: util.py
@author: zj
@description: 
"""

import cv2
from typing import Any

import numpy as np
from PIL import Image


def default_loader(path: str, rgb=False) -> Any:
    image = cv2.imread(path)
    if rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def default_converter(img: Image.Image, rgb=False) -> Any:
    image = np.array(img)
    if not rgb:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

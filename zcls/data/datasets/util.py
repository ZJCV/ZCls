# -*- coding: utf-8 -*-

"""
@date: 2021/10/4 下午1:26
@file: util.py
@author: zj
@description: 
"""

import cv2
from typing import Any


def default_loader(path: str, rgb=True) -> Any:
    image = cv2.imread(path)
    if rgb:
        cv2.cvtColor(image, image, cv2.COLOR_BGR2RGB)
    return image

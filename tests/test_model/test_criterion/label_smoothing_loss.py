# -*- coding: utf-8 -*-

"""
@date: 2021/6/7 下午5:11
@file: label_smoothing_loss.py
@author: zj
@description: 
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.loss import _WeightedLoss
from zcls.model.criterions.label_smoothing_loss import LabelSmoothingLoss

if __name__ == "__main__":
    # 1. Devin Yang
    crit = LabelSmoothingLoss(classes=5, smoothing=0.5)
    predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                                 [0, 0.9, 0.2, 0.2, 1],
                                 [1, 0.2, 0.7, 0.9, 1]])
    v = crit(predict,
             torch.LongTensor([2, 1, 0]))
    print(v)

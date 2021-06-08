# -*- coding: utf-8 -*-

"""
@date: 2020/12/21 下午8:17
@file: label_smoothing_loss.py
@author: zj
@description: 
"""

from abc import ABC
import torch.nn as nn
import torch.nn.functional as F

from .. import registry
from zcls.config.key_word import KEY_OUTPUT, KEY_LOSS


@registry.CRITERION.register('LabelSmoothingLoss')
class LabelSmoothingLoss(nn.Module, ABC):
    """
    Label smoothing cross entropy loss
    Refer to:
    1. [解决过拟合：如何在PyTorch中使用标签平滑正则化](https://zhuanlan.zhihu.com/p/123077402)
    2. [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)
    3. [[译]Rethinking the Inception Architecture for Computer Vision](https://blog.zhujian.life/posts/a0a2be91.html)
    4. [Label Smoothing in PyTorch](https://stackoverflow.com/questions/55681502/label-smoothing-in-pytorch)
    """

    def __init__(self, cfg):
        super(LabelSmoothingLoss, self).__init__()
        self.epsilon = cfg.MODEL.CRITERION.SMOOTHING
        self.reduction = cfg.MODEL.CRITERION.REDUCTION

    def __call__(self, output_dict, targets):
        assert isinstance(output_dict, dict) and KEY_OUTPUT in output_dict.keys()
        inputs = output_dict[KEY_OUTPUT]

        n = inputs.size()[-1]
        log_preds = F.log_softmax(inputs, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1), reduction=self.reduction)
        nll = F.nll_loss(log_preds, targets, reduction=self.reduction)

        return {KEY_LOSS: self.linear_combination(loss / n, nll, self.epsilon)}

    def reduce_loss(self, loss, reduction='mean'):
        return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss

    def linear_combination(self, x, y, epsilon):
        return epsilon * x + (1 - epsilon) * y

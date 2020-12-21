# -*- coding: utf-8 -*-

"""
@date: 2020/12/21 下午8:17
@file: label_smoothing_loss.py
@author: zj
@description: 
"""

import torch.nn as nn
import torch.nn.functional as F
from .. import registry


@registry.CRITERION.register('LabelSmoothingLoss')
class LabelSmoothingLoss(nn.Module):
    """
    标签平滑交叉熵损失
    参考：
    [解决过拟合：如何在PyTorch中使用标签平滑正则化](https://zhuanlan.zhihu.com/p/123077402)
    [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)
    [[译]Rethinking the Inception Architecture for Computer Vision](https://blog.zhujian.life/posts/a0a2be91.html)
    """

    def __init__(self, cfg):
        super(LabelSmoothingLoss, self).__init__()
        self.epsilon = 0.1
        self.reduction = 'mean'

    def __call__(self, output_dict, targets):
        assert isinstance(output_dict, dict) and 'probs' in output_dict.keys()
        inputs = output_dict['probs']

        n = inputs.size()[-1]
        log_preds = F.log_softmax(inputs, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1), reduction=self.reduction)
        nll = F.nll_loss(log_preds, targets, reduction=self.reduction)

        return {'loss': self.linear_combination(loss / n, nll, self.epsilon)}

    def reduce_loss(self, loss, reduction='mean'):
        return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss

    def linear_combination(self, x, y, epsilon):
        return epsilon * x + (1 - epsilon) * y

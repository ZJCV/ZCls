# -*- coding: utf-8 -*-

"""
@date: 2022/4/4 下午10:13
@file: large_margin_softmax_loss.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn

'''
proposed in the BMVC2019 paper: [Large Margin in Softmax Cross-Entropy Loss
link to paper](https://staff.aist.go.jp/takumi.kobayashi/publication/2019/BMVC2019.pdf)
'''


##
# version 1: use torch.autograd
class LargeMarginSoftmaxV1(nn.Module):

    def __init__(self, lam=0.3, reduction='mean', ignore_index=255):
        super(LargeMarginSoftmaxV1, self).__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.lam = lam
        self.ce_crit = nn.CrossEntropyLoss(
            reduction='none', ignore_index=ignore_index)

    def __call__(self, logits, label):
        # def forward(self, logits, label):
        '''
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LargeMarginSoftmaxV1()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        '''
        # overcome ignored label
        logits = logits.float()
        if logits.requires_grad is True:
            logits.retain_grad()
            logits.register_hook(lambda grad: grad)
        with torch.no_grad():
            num_classes = logits.size(1)
            coeff = 1. / (num_classes - 1.)
            lb = label.clone().detach()
            mask = label == self.ignore_index
            lb[mask] = 0
            idx = torch.zeros_like(logits).scatter_(1, lb.unsqueeze(1), 1.)

        lgts = logits - idx * 1.e6
        q = lgts.softmax(dim=1)
        q = q * (1. - idx)

        log_q = lgts.log_softmax(dim=1)
        log_q = log_q * (1. - idx)
        mg_loss = ((q - coeff) * log_q) * (self.lam / 2)
        mg_loss = mg_loss * (1. - idx)
        mg_loss = mg_loss.sum(dim=1)

        ce_loss = self.ce_crit(logits, label)
        loss = ce_loss + mg_loss
        loss = loss[mask == 0]

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss


def build_large_margin_softmax_loss(args):
    return LargeMarginSoftmaxV1()

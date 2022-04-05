# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午1:37
@file: build.py
@author: zj
@description: 
"""

from .multi_step_lr import build_multistep_lr
from .cosine_annealing_lr import build_cosine_annearling_lr


def adjust_learning_rate(args, optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    # factor = epoch // 30
    #
    # if epoch >= 80:
    #     factor = factor + 1
    #
    # lr = args.lr * (0.1 ** factor)
    lr = args.lr

    """Warmup"""
    if epoch < 5:
        lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)

    # if(args.local_rank == 0):
    #     print("epoch = {}, step = {}, lr = {}".format(epoch, step, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def build_lr_scheduler(args, optimizer):
    if args.lr_scheduler == 'MultiStepLR':
        return build_multistep_lr(args, optimizer)
    elif args.lr_scheduler == 'CosineAnnealingLR':
        return build_cosine_annearling_lr(args, optimizer)
    else:
        raise ValueError(f'{args.lr_scheduler} do not support')

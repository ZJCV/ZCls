# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午1:37
@file: build.py
@author: zj
@description: 
"""


# def adjust_learning_rate(lr, optimizer, epoch, step, len_epoch):
def adjust_learning_rate(args, optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 1

    lr = args.lr * (0.1 ** factor)
    # lr = lr * (0.1 ** factor)

    """Warmup"""
    if epoch < 5:
        lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)

    # if(args.local_rank == 0):
    #     print("epoch = {}, step = {}, lr = {}".format(epoch, step, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

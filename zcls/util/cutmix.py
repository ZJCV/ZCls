# -*- coding: utf-8 -*-

"""
@date: 2021/7/26 下午10:10
@file: cutmix.py
@author: zj
@description:
refer to [ clovaai/CutMix-PyTorch](https://github.com/clovaai/CutMix-PyTorch)
"""

import torch

import numpy as np

from zcls.config.key_word import KEY_LOSS


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(images, targets, alpha=1.0, device=torch.device('cpu')):
    '''
    Returns mixed inputs, pairs of targets, and lambda
    '''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = images.size()[0]
    rand_index = torch.randperm(batch_size).to(device)

    targets_a = targets
    targets_b = targets[rand_index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
    images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))

    return images, targets_a, targets_b, lam


def cutmix_criterion(criterion, output_dict: dict, targets_a: torch.Tensor, targets_b: torch.Tensor, lam):
    loss_a = criterion(output_dict, targets_a)[KEY_LOSS]
    loss_b = criterion(output_dict, targets_b)[KEY_LOSS]

    total_loss = lam * loss_a + (1 - lam) * loss_b

    return {KEY_LOSS: total_loss}


def cutmix_evaluate(evaluator, output_dict, targets_a, targets_b, lam):
    acc_dict_a = evaluator.evaluate_train(output_dict, targets_a)
    acc_dict_b = evaluator.evaluate_train(output_dict, targets_b)

    total_acc_dict = dict()

    for (a_key, a_value), (b_key, b_value) in zip(acc_dict_a.items(), acc_dict_b.items()):
        assert a_key == b_key
        total_acc_dict[a_key] = lam * a_value + (1 - lam) * b_value

    return total_acc_dict

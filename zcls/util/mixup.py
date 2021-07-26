# -*- coding: utf-8 -*-

"""
@date: 2021/7/26 上午10:45
@file: mixup.py
@author: zj
@description:
refer to [facebookresearch/mixup-cifar10](https://github.com/facebookresearch/mixup-cifar10)
"""

import torch

import numpy as np

from zcls.config.key_word import KEY_LOSS


def mixup_data(images, targets, alpha=1.0, device=torch.device('cpu')):
    '''
    Returns mixed inputs, pairs of targets, and lambda
    '''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = images.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * images + (1 - lam) * images[index, :]
    targets_a, targets_b = targets, targets[index]
    return mixed_x, targets_a, targets_b, lam


def mixup_criterion(criterion, output_dict: dict, targets_a: torch.Tensor, targets_b: torch.Tensor, lam):
    loss_a = criterion(output_dict, targets_a)[KEY_LOSS]
    loss_b = criterion(output_dict, targets_b)[KEY_LOSS]

    total_loss = lam * loss_a + (1 - lam) * loss_b

    return {KEY_LOSS: total_loss}


def mixup_evaluate(evaluator, output_dict, targets_a, targets_b, lam):
    acc_dict_a = evaluator.evaluate_train(output_dict, targets_a)
    acc_dict_b = evaluator.evaluate_train(output_dict, targets_b)

    total_acc_dict = dict()

    for (a_key, a_value), (b_key, b_value) in zip(acc_dict_a.items(), acc_dict_b.items()):
        assert a_key == b_key
        total_acc_dict[a_key] = lam * a_value + (1 - lam) * b_value

    return total_acc_dict

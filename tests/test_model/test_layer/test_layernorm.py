# -*- coding: utf-8 -*-

"""
@date: 2020/11/26 下午8:22
@file: test_layernorm.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn

from zcls.model.layers.layer_norm import LayerNorm


def compare_bn(bn1, bn2):
    err = False

    if bn1.elementwise_affine and bn2.elementwise_affine:
        if not torch.allclose(bn1.weight, bn2.weight):
            print('Diff in weight: {} vs {}'.format(
                bn1.weight, bn2.weight))
            err = True

        if not torch.allclose(bn1.bias, bn2.bias):
            print('Diff in bias: {} vs {}'.format(
                bn1.bias, bn2.bias))
            err = True

    if not err:
        print('All parameters are equal!')


# Init BatchNorm layers
my_bn = LayerNorm([3, 100, 100], elementwise_affine=True)
bn = nn.LayerNorm([3, 100, 100], elementwise_affine=True)

compare_bn(my_bn, bn)  # weight and bias should be different
# Load weight and bias
my_bn.load_state_dict(bn.state_dict())
compare_bn(my_bn, bn)

print('train...')

# Run train
for _ in range(10):
    scale = torch.randint(1, 10, (1,)).float()
    bias = torch.randint(-10, 10, (1,)).float()
    x = torch.randn(10, 3, 100, 100) * scale + bias
    out1 = my_bn(x)
    out2 = bn(x)
    compare_bn(my_bn, bn)

    if torch.allclose(out1, out2):
        print('Max diff: ', (out1 - out2).abs().max())

print('test...')

# Run eval
my_bn.eval()
bn.eval()
for _ in range(10):
    scale = torch.randint(1, 10, (1,)).float()
    bias = torch.randint(-10, 10, (1,)).float()
    x = torch.randn(10, 3, 100, 100) * scale + bias
    out1 = my_bn(x)
    out2 = bn(x)
    compare_bn(my_bn, bn)

    if torch.allclose(out1, out2):
        print('Max diff: ', (out1 - out2).abs().max())

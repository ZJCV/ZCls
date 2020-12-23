# -*- coding: utf-8 -*-

"""
@date: 2020/12/23 下午2:38
@file: test_group_weight.py
@author: zj
@description: 
"""

import torch.nn as nn

from zcls.optim.optimizers.build import group_weight


class TestA(nn.Module):

    def __init__(self):
        super(TestA, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(5)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Sequential(
            nn.Conv2d(5, 10, kernel_size=1, stride=1),
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Linear(10, 20)


def test_group_weight():
    model = TestA()
    print(model)

    groups = group_weight(model)
    print(groups)


if __name__ == '__main__':
    test_group_weight()

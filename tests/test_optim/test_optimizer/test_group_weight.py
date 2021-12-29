# -*- coding: utf-8 -*-

"""
@date: 2020/12/23 下午2:38
@file: test_group_weight.py
@author: zj
@description: 
"""

import torch.nn as nn

from zcls.config import cfg
from zcls.optim.optimizers.build import filter_weight


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

    groups = filter_weight(cfg, model)
    print(groups)

    import torch.optim as optim
    optimizer = optim.SGD(groups, lr=1e-3, momentum=0.9, weight_decay=0.4)
    optimizer.step()


if __name__ == '__main__':
    test_group_weight()

# -*- coding: utf-8 -*-

"""
@date: 2020/11/11 下午7:09
@file: compute_mean_std.py
@author: zj
@description: 计算cifar100的mean以及std
"""

import torch
from tqdm import tqdm
from torchvision.datasets.cifar import CIFAR100
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


def main():
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    data_set = CIFAR100('./data/cifar', train=True, transform=transform, download=True)
    data_loader = DataLoader(data_set, batch_size=24, num_workers=8, shuffle=False)

    nb_samples = 0.
    channel_mean = torch.zeros(3)
    channel_std = torch.zeros(3)
    for images, targets in tqdm(data_loader):
        # scale image to be between 0 and 1
        N, C, H, W = images.shape[:4]
        data = images.view(N, C, -1)

        channel_mean += data.mean(2).sum(0)
        channel_std += data.std(2).sum(0)
        nb_samples += N

    channel_mean /= nb_samples
    channel_std /= nb_samples
    print(channel_mean, channel_std)


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-

"""
@date: 2021/9/24 上午11:38
@file: rank_dataset.py
@author: zj
@description: 
"""

import os
import math
import json

from PIL import Image

import torch
from torch.utils.data import Dataset
from .evaluator.general_evaluator import GeneralEvaluator


def get_base_info(json_path):
    assert os.path.isfile(json_path), json_path
    with open(json_path, 'r') as f:
        data_dict = json.load(f)

    classes = list(data_dict.keys())
    for key in classes:
        img_list = data_dict[key]

        for img_path in img_list:
            assert os.path.isfile(img_path), img_path
    data_dict.clear()

    return classes


def get_total_data(json_path, classes):
    assert os.path.isfile(json_path)
    assert isinstance(classes, list)

    total_img_list = list()
    total_label_list = list()

    with open(json_path, 'r') as f:
        data_dict = json.load(f)

    for key in classes:
        img_list = data_dict[key]

        label = classes.index(key)
        for img_path in img_list:
            assert os.path.isfile(img_path), img_path
            total_img_list.append(img_path)
            total_label_list.append(label)

    return total_img_list, total_label_list


def get_rank_data(seed, epoch, rank, num_replicas,
                  json_path, classes):
    total_img_list, total_label_list = get_total_data(json_path, classes)
    total_size = len(total_img_list)

    # deterministically shuffle based on epoch and seed
    g = torch.Generator()
    g.manual_seed(seed + epoch)
    indices = torch.randperm(total_size, generator=g).tolist()  # type: ignore[arg-type]

    # Split to nearest available length that is evenly divisible.
    # This is to ensure each rank receives the same amount of data when
    # using this Sampler.
    num_samples = math.ceil(
        # `type:ignore` is required because Dataset cannot provide a default __len__
        # see NOTE in pytorch/torch/utils/data/sampler.py
        (total_size - num_replicas) / num_replicas  # type: ignore[arg-type]
    )

    new_total_size = num_replicas * num_samples
    new_indices = indices[:new_total_size]
    # subsample
    dst_indices = new_indices[rank:new_total_size:num_replicas]
    assert len(dst_indices) == num_samples

    sub_img_list = list()
    sub_label_list = list()
    for idx in dst_indices:
        sub_img_list.append(total_img_list[idx])
        sub_label_list.append(total_label_list[idx])

    return sub_img_list, sub_label_list


class RankDataset(Dataset):
    """
    refer to [RankDataset：超大规模数据集加载利器 [炼丹炉番外篇-1]](https://zhuanlan.zhihu.com/p/357809861)
    """

    def __init__(self, root, transform=None, target_transform=None, top_k=(1, 5),
                 world_size: int = 1, rank_id: int = 0, seed: int = 0, epoch: int = 0):
        assert world_size > 1
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.seed = seed
        self.rank = rank_id
        self.num_replicas = world_size

        self.classes = get_base_info(root)
        self.set_epoch(epoch)

        self._update_evaluator(top_k)

    def __getitem__(self, index: int):
        img_path = self.img_list[index]
        target = self.label_list[index]

        image = Image.open(img_path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self) -> int:
        return self.length

    def _update_evaluator(self, top_k):
        self.evaluator = GeneralEvaluator(self.classes, top_k=top_k)

    def get_classes(self):
        return self.classes

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.root + ')'

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
        self.img_list, self.label_list = get_rank_data(self.seed, self.epoch, self.rank, self.num_replicas,
                                                       self.root, self.classes)
        assert len(self.img_list) == len(self.label_list)
        self.length = len(self.img_list)

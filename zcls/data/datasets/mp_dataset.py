# -*- coding: utf-8 -*-

"""
@date: 2021/9/27 下午7:43
@file: mp_dataset.py
@author: zj
@description: 
"""

import os
import json

from PIL import Image

import torch
from torch.utils.data import IterableDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler, SequentialSampler

import zcls.util.distributed as du

from .evaluator.general_evaluator import GeneralEvaluator


def get_base_info(json_path):
    assert os.path.isfile(json_path), json_path
    with open(json_path, 'r') as f:
        data_dict = json.load(f)

    length = 0
    classes = list(data_dict.keys())
    for key in classes:
        img_list = data_dict[key]

        for img_path in img_list:
            assert os.path.isfile(img_path), img_path
        length += len(img_list)
    data_dict.clear()

    return classes, length


def build_sampler(dataset, num_gpus=1, random_sample=False):
    world_size = du.get_world_size()
    rank = du.get_rank()

    if num_gpus <= 1:
        if random_sample:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
    else:
        shuffle = random_sample
        sampler = DistributedSampler(dataset,
                                     num_replicas=world_size,
                                     rank=rank,
                                     shuffle=shuffle)

    return sampler


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


def get_subdata(json_path, classes, indices):
    total_img_list, total_label_list = get_total_data(json_path, classes)

    sub_img_list = list()
    sub_label_list = list()
    for idx in indices:
        sub_img_list.append(total_img_list[idx])
        sub_label_list.append(total_label_list[idx])

    return sub_img_list, sub_label_list


def shuffle_dataset(sampler, cur_epoch, is_shuffle=False):
    """"
    Shuffles the data.
    Args:
        loader (loader): data loader to perform shuffle.
        cur_epoch (int): number of the current epoch.
        is_shuffle (bool): need to shuffle the data
    """
    if not is_shuffle:
        return
    assert isinstance(
        sampler, (RandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        sampler.set_epoch(cur_epoch)


class MPDataset(IterableDataset):

    def __init__(self, root, transform=None, target_transform=None, top_k=(1, 5), shuffle: bool = False,
                 num_gpus: int = 1, rank_id: int = 0, epoch: int = 0):
        super(MPDataset).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.rank = rank_id
        self.num_replicas = num_gpus

        self.classes, self.length = get_base_info(root)
        self.sampler = build_sampler(self, self.num_replicas, shuffle)
        self.set_epoch(epoch)

        self._update_evaluator(top_k)

    def parse_file(self, img_list, label_list):
        for img_path, target in zip(img_list, label_list):
            image = Image.open(img_path)
            if self.transform is not None:
                image = self.transform(image)
            if self.target_transform is not None:
                target = self.target_transform(target)

            yield image, target

    def get_indices(self):
        indices = list(self.sampler)
        length = len(indices)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # in a worker process
            worker_id = worker_info.id
            # split workload
            indices = indices[worker_id:length:worker_info.num_workers]
        # single-process data loading, return the full iterator

        return indices

    def __iter__(self):
        indices = self.get_indices()
        img_list, label_list = get_subdata(self.root, self.classes, indices)
        assert len(img_list) == len(label_list)

        return iter(self.parse_file(img_list, label_list))

    def __len__(self):
        return self.length

    def _update_evaluator(self, top_k):
        self.evaluator = GeneralEvaluator(self.classes, top_k=top_k)

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
        shuffle_dataset(self.sampler, self.epoch, self.shuffle)

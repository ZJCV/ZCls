# -*- coding: utf-8 -*-

"""
@date: 2021/9/27 下午7:43
@file: mp_dataset.py
@author: zj
@description: see [zjykzj/MPDataset](https://github.com/zjykzj/MPDataset)
"""

import os

import torch
from torch.utils.data import IterableDataset
from torch.utils.data import RandomSampler, SequentialSampler

from zcls.config.key_word import KEY_DATASET, KEY_CLASSES, KEY_SEP
from ..samplers.distributed_sampler import DistributedSampler
from .evaluator.general_evaluator import GeneralEvaluator
from .util import default_loader


def get_base_info(cls_path, data_path):
    assert os.path.isfile(cls_path), cls_path
    classes = list()
    with open(cls_path, 'r') as f:
        for idx, line in enumerate(f):
            classes.append(line.strip())

    assert os.path.isfile(data_path), data_path
    length = 0
    with open(data_path, 'r') as f:
        for _ in f:
            length += 1
    return classes, length


def build_sampler(dataset, num_gpus=1, random_sample=False, rank_id=0, drop_last=False):
    if num_gpus <= 1:
        if random_sample:
            # different work use same generator
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
            sampler = RandomSampler(dataset, generator=generator)
        else:
            sampler = SequentialSampler(dataset)
    else:
        shuffle = random_sample
        # using dataset.length replace dataset
        sampler = DistributedSampler(dataset.length,
                                     num_replicas=num_gpus,
                                     rank=rank_id,
                                     shuffle=shuffle,
                                     drop_last=drop_last)

    return sampler


def get_subset_data(data_path, indices):
    sub_img_list = [0 for _ in indices]
    sub_label_list = [0 for _ in indices]

    idx_dict = dict()
    for i, idx in enumerate(indices):
        idx_dict[idx] = i

    indices_set = set(indices)
    with open(data_path, 'r') as f:
        for idx, line in enumerate(f):
            if idx in indices_set:
                img_path, target = line.strip().split(KEY_SEP)[:2]

                list_idx = idx_dict[idx]
                sub_img_list[list_idx] = img_path
                sub_label_list[list_idx] = int(target)

    return sub_img_list, sub_label_list


def shuffle_dataset(sampler, cur_epoch, is_shuffle=False):
    """"
    Shuffles the data.
    Args:
        sampler (sampler): sampler to perform shuffle.
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

    def __init__(self, root, transform=None, target_transform=None, top_k=(1, 5), keep_rgb: bool = False,
                 shuffle: bool = False, num_gpus: int = 1, rank_id: int = 0, epoch: int = 0, drop_last: bool = False):
        super(MPDataset).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.keep_rgb = keep_rgb
        self.shuffle = shuffle

        self.rank = rank_id
        self.num_replicas = num_gpus

        self.data_path = os.path.join(self.root, KEY_DATASET)
        self.cls_path = os.path.join(self.root, KEY_CLASSES)
        self.classes, self.length = get_base_info(self.cls_path, self.data_path)
        self.sampler = build_sampler(self, num_gpus=self.num_replicas, random_sample=self.shuffle,
                                     rank_id=self.rank, drop_last=drop_last)
        self.set_epoch(epoch)

        self._update_evaluator(top_k)

    def parse_file(self, img_list, label_list):
        for img_path, target in zip(img_list, label_list):
            image = default_loader(img_path, rgb=self.keep_rgb)
            if self.transform is not None:
                image = self.transform(image)
            if self.target_transform is not None:
                target = self.target_transform(target)

            yield image, target

    def get_indices(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            # in a worker process
            worker_id = worker_info.id
            # split workload
            indices = self.indices[worker_id:self.indices_length:worker_info.num_workers]
        else:
            # single-process data loading, return the full iterator
            indices = self.indices

        return indices

    def __iter__(self):
        indices = self.get_indices()

        img_list, label_list = get_subset_data(self.data_path, indices)
        assert len(img_list) == len(label_list)

        return iter(self.parse_file(img_list, label_list))

    def __len__(self):
        return self.indices_length if self.num_replicas > 1 else self.length

    def _update_evaluator(self, top_k):
        self.evaluator = GeneralEvaluator(self.classes, top_k=top_k)

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler.

        Args:
            epoch (int): Epoch number.
        """
        shuffle_dataset(self.sampler, epoch, self.shuffle)
        self.indices = list(self.sampler)
        self.indices_length = len(self.indices)

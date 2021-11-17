# -*- coding: utf-8 -*-

"""
@date: 2021/9/23 下午10:06
@file: general_dataset_v2.py
@author: zj
@description: 
"""

import os
import json

from torch.utils.data import Dataset

from .evaluator.general_evaluator import GeneralEvaluator
from .util import default_loader


class GeneralDatasetV2(Dataset):

    def __init__(self, root, transform=None, target_transform=None, top_k=(1, 5), keep_rgb=False):
        assert os.path.isfile(root)
        with open(root, 'r') as f:
            data_dict = json.load(f)

        self.classes = list(data_dict.keys())
        self.total_img_list = list()
        self.total_label_list = list()
        for key in self.classes:
            img_list = data_dict[key]

            label = self.classes.index(key)
            for img_path in img_list:
                assert os.path.isfile(img_path), img_path

                self.total_img_list.append(img_path)
                self.total_label_list.append(label)

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.keep_rgb = keep_rgb
        self._update_evaluator(top_k)

    def __getitem__(self, index: int):
        img_path = self.total_img_list[index]
        target = self.total_label_list[index]

        image = default_loader(img_path, rgb=self.keep_rgb)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self) -> int:
        return len(self.total_img_list)

    def _update_evaluator(self, top_k):
        self.evaluator = GeneralEvaluator(self.classes, top_k=top_k)

    def get_classes(self):
        return self.classes

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.root + ')'

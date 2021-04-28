# -*- coding: utf-8 -*-

"""
@date: 2021/2/23 下午8:22
@file: imagenet.py
@author: zj
@description: 
"""

import json

from .lmdb_dataset import LMDBDataset


class LMDBImageNet(LMDBDataset):
    """
    [What is the meta.bin file used by the ImageNet dataset? #1646](https://github.com/pytorch/vision/issues/1646)
    torchvision will parse the devkit archive of the ImageNet2012 classification dataset and save
    the meta information in a binary file.
    about problem: TypeError: can't pickle Environment objects
    refert to [TypeError: can't pickle Environment objects when num_workers > 0 for LSUN #689](https://github.com/pytorch/vision/issues/689)
    """

    def __init__(self, root, transform=None, target_transform=None, top_k=(1, 5)):
        super().__init__(root, transform, target_transform, top_k)

    def get_classes(self):
        """
        refert to [Imagenet classes](https://discuss.pytorch.org/t/imagenet-classes/4923)
        :return:
        """
        idx2label = []
        # cls2label = {}
        with open("zcls/assets/imagenet_class_index.json", "r") as read_file:
            class_idx = json.load(read_file)
            idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
            # cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}
        return idx2label

# -*- coding: utf-8 -*-

"""
@date: 2021/10/4 上午9:40
@file: augment.py
@author: zj
@description: 
"""
import numpy as np
from PIL import Image

import torchvision.transforms.autoaugment as autoaugment
from torchvision.transforms.autoaugment import AutoAugmentPolicy


class AutoAugment(object):
    """
    AutoAugment data augmentation method based on
    `"AutoAugment: Learning Augmentation Strategies from Data" <https://arxiv.org/pdf/1805.09501.pdf>`_.

    Args:
        policy (AutoAugmentPolicy): Desired policy enum defined by
            :class:`torchvision.transforms.autoaugment.AutoAugmentPolicy`. Default is ``AutoAugmentPolicy.IMAGENET``.
        p (float): probability of applying the transform. Default: 0.5.
    """

    def __init__(self, policy: AutoAugmentPolicy = AutoAugmentPolicy.IMAGENET, p=0.5):
        self.policy = policy
        self.p = p

        self.t = autoaugment.AutoAugment(policy=self.policy)

    def __call__(self, image):
        if np.random.rand(1) < self.p:
            image = np.array(self.t(Image.fromarray(image)))

        return image

    def __repr__(self):
        return self.__class__.__name__ + '(policy={0}, p={1})'.format(self.policy, self.p)

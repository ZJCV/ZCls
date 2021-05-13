# -*- coding: utf-8 -*-

"""
@date: 2021/2/2 下午5:46
@file: test_regvgg.py
@author: zj
@description: 
"""

import torch

from zcls.config import cfg
from zcls.config.key_word import KEY_OUTPUT
from zcls.model.recognizers.build import build_recognizer
from zcls.model.recognizers.vgg.repvgg import RepVGG
from zcls.model.backbones.vgg.repvgg_backbone import arch_settings
from zcls.model.conv_helper import insert_repvgg_block, insert_acblock, fuse_repvgg_block, fuse_acblock


def test_regvgg():
    data = torch.randn(1, 3, 224, 224)
    for key in arch_settings.keys():
        print('*' * 10, key)
        cfg.merge_from_file('configs/cifar/r50_cifar100_224_e100_rmsprop.yaml')
        model = RepVGG(cfg)
        # print(model)
        outputs = model(data)[KEY_OUTPUT]
        assert outputs.shape == (1, 100)

        print('insert_regvgg_block -> fuse_regvgg_block')
        insert_repvgg_block(model)
        # print(model)
        model.eval()
        outputs_insert = model(data)[KEY_OUTPUT]
        fuse_repvgg_block(model)
        # print(model)
        model.eval()
        outputs_fuse = model(data)[KEY_OUTPUT]

        # print(outputs_insert)
        # print(outputs_fuse)
        print(torch.sqrt(torch.sum((outputs_insert - outputs_fuse) ** 2)))
        print(torch.allclose(outputs_insert, outputs_fuse, atol=1e-8))
        assert torch.allclose(outputs_insert, outputs_fuse, atol=1e-8)

        print('insert_regvgg_block -> insert_acblock -> fuse_acblock -> fuse_regvgg_block')
        insert_repvgg_block(model)
        insert_acblock(model)
        # print(model)
        model.eval()
        outputs_insert = model(data)[KEY_OUTPUT]
        fuse_acblock(model)
        fuse_repvgg_block(model)
        # print(model)
        model.eval()
        outputs_fuse = model(data)[KEY_OUTPUT]

        print(torch.sqrt(torch.sum((outputs_insert - outputs_fuse) ** 2)))
        print(torch.allclose(outputs_insert, outputs_fuse, atol=1e-6))
        assert torch.allclose(outputs_insert, outputs_fuse, atol=1e-6)

        print('insert_acblock -> insert_regvgg_block -> fuse_regvgg_block -> fuse_acblock')
        insert_repvgg_block(model)
        insert_acblock(model)
        # print(model)
        model.eval()
        outputs_insert = model(data)[KEY_OUTPUT]
        fuse_acblock(model)
        fuse_repvgg_block(model)
        # print(model)
        model.eval()
        outputs_fuse = model(data)[KEY_OUTPUT]

        print(torch.sqrt(torch.sum((outputs_insert - outputs_fuse) ** 2)))
        print(torch.allclose(outputs_insert, outputs_fuse, atol=1e-6))
        assert torch.allclose(outputs_insert, outputs_fuse, atol=1e-6)


if __name__ == '__main__':
    test_regvgg()

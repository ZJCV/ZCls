# -*- coding: utf-8 -*-

"""
@date: 2021/5/4 下午7:11
@file: torchvision_resnet_to_zcls_resnet.py
@author: zj
@description: Transform torchvision pretrained model into zcls format
first, download RepVGG repo and set env
git clone https://github.com/DingXiaoH/RepVGG.git
export PYTHONPATH=$PYTHONPATH:/path/to/RepVGG
"""

import os
import torch
from repvgg import create_RepVGG_A0, create_RepVGG_A1, create_RepVGG_A2, create_RepVGG_B0, create_RepVGG_B1, \
    create_RepVGG_B1g2, create_RepVGG_B1g4, create_RepVGG_B2, create_RepVGG_B2g4, create_RepVGG_B3, create_RepVGG_B3g4

from zcls.model.recognizers.build import build_recognizer
from zcls.util.checkpoint import CheckPointer
from zcls.config import cfg
from zcls.model.conv_helper import fuse_repvgg_block


def convert(official_model, zcls_model):
    official_dict = official_model.state_dict()
    zcls_dict = zcls_model.state_dict()

    for (o_k, o_v), (z_k, z_v) in zip(official_dict.items(), zcls_dict.items()):
        # print(o_k, o_v.shape, z_k, z_v.shape)
        zcls_dict[z_k] = o_v

    return zcls_dict


def process(item, cfg_file):
    if item == 'repvgg_a0':
        official_model = create_RepVGG_A0()
        official_model.load_state_dict(torch.load('/home/zj/repos/RepVGG/RepVGG-A0-train.pth'))
    elif item == 'repvgg_a1':
        official_model = create_RepVGG_A1()
        official_model.load_state_dict(torch.load('/home/zj/repos/RepVGG/RepVGG-A1-train.pth'))
    elif item == 'repvgg_a2':
        official_model = create_RepVGG_A2()
        official_model.load_state_dict(torch.load('/home/zj/repos/RepVGG/RepVGG-A2-train.pth'))
    elif item == 'repvgg_b0':
        official_model = create_RepVGG_B0()
        official_model.load_state_dict(torch.load('/home/zj/repos/RepVGG/RepVGG-B0-train.pth'))
    elif item == 'repvgg_b1':
        official_model = create_RepVGG_B1()
        official_model.load_state_dict(torch.load('/home/zj/repos/RepVGG/RepVGG-B1-train.pth'))
    elif item == 'repvgg_b1g2':
        official_model = create_RepVGG_B1g2()
        official_model.load_state_dict(torch.load('/home/zj/repos/RepVGG/RepVGG-B1g2-train.pth'))
    elif item == 'repvgg_b1g4':
        official_model = create_RepVGG_B1g4()
        official_model.load_state_dict(torch.load('/home/zj/repos/RepVGG/RepVGG-B1g4-train.pth'))
    elif item == 'repvgg_b2':
        official_model = create_RepVGG_B2()
        official_model.load_state_dict(torch.load('/home/zj/repos/RepVGG/RepVGG-B2-train.pth'))
    elif item == 'repvgg_b2g4':
        official_model = create_RepVGG_B2g4()
        official_model.load_state_dict(torch.load('/home/zj/repos/RepVGG/RepVGG-B2g4-200epochs-train.pth'))
    elif item == 'repvgg_b3':
        official_model = create_RepVGG_B3()
        official_model.load_state_dict(torch.load('/home/zj/repos/RepVGG/RepVGG-B3-200epochs-train.pth'))
    elif item == 'repvgg_b3g4':
        official_model = create_RepVGG_B3g4()
        official_model.load_state_dict(torch.load('/home/zj/repos/RepVGG/RepVGG-B3g4-200epochs-train.pth'))
    else:
        raise ValueError(f"{item} doesn't exists")

    cfg.merge_from_file(cfg_file)
    zcls_model = build_recognizer(cfg, torch.device('cpu'))
    zcls_model_dict = convert(official_model, zcls_model)
    zcls_model.load_state_dict(zcls_model_dict)

    res_dir = 'outputs/converters/'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    checkpoint = CheckPointer(model=zcls_model, save_dir=res_dir, save_to_disk=True)
    checkpoint.save(f'{item}_train_imagenet')

    fuse_repvgg_block(zcls_model)
    checkpoint = CheckPointer(model=zcls_model, save_dir=res_dir, save_to_disk=True)
    checkpoint.save(f'{item}_infer_imagenet')


if __name__ == '__main__':
    item_list = ['repvgg_a0', 'repvgg_a1', 'repvgg_a2', 'repvgg_b0', 'repvgg_b1', 'repvgg_b1g2', 'repvgg_b1g4',
                 'repvgg_b2', 'repvgg_b2g4', 'repvgg_b3', 'repvgg_b3g4']
    cfg_file_list = [
        'repvgg_a0_train_zcls_imagenet_224.yaml',
        'repvgg_a1_train_zcls_imagenet_224.yaml',
        'repvgg_a2_train_zcls_imagenet_224.yaml',
        'repvgg_b0_train_zcls_imagenet_224.yaml',
        'repvgg_b1_train_zcls_imagenet_224.yaml',
        'repvgg_b1g2_train_zcls_imagenet_224.yaml',
        'repvgg_b1g4_train_zcls_imagenet_224.yaml',
        'repvgg_b2_train_zcls_imagenet_224.yaml',
        'repvgg_b2g4_train_zcls_imagenet_224.yaml',
        'repvgg_b3_train_zcls_imagenet_224.yaml',
        'repvgg_b3g4_train_zcls_imagenet_224.yaml',
    ]
    prefix_path = 'configs/benchmarks/repvgg'
    for item, cfg_file in zip(item_list, cfg_file_list):
        config_path = os.path.join(prefix_path, cfg_file)
        print(item, config_path)
        process(item, config_path)

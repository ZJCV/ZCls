# -*- coding: utf-8 -*-

"""
@date: 2021/5/4 下午7:11
@file: official_ghostnet_to_zcls_ghostnet.py
@author: zj
@description: Transform official pretrained model into zcls format
first, download GhostNet repo and set env

git clone https://github.com/huawei-noah/CV-Backbones.git

export PYTHONPATH=$PYTHONPATH:/path/to/CV-Backbones

pretrained model in /path/to/
"""

import os
import torch
import ghostnet
from zcls.model.recognizers.ghostnet.ghostnet import GhostNet
from zcls.util.checkpoint import CheckPointer
from zcls.config import cfg


def convert(official_model, zcls_model):
    official_dict = official_model.state_dict()
    zcls_dict = zcls_model.state_dict()

    v_list = [v for k, v in official_dict.items()]
    v_list = v_list[:6] + v_list[-10:-4] + v_list[6:-10] + v_list[-4:]

    for t_v, (z_k, z_v) in zip(v_list, zcls_dict.items()):
        zcls_dict[z_k] = t_v.reshape(z_v.shape)

    return zcls_dict


def process(item, cfg_file):
    if item == 'ghostnet_x1_0':
        official_model = ghostnet.ghostnet()
        official_model.eval()

        state_dict = torch.load('/home/zj/PycharmProjects/CV-Backbones/ghostnet_pytorch/models/state_dict_73.98.pth', map_location=torch.device('cpu'))
        official_model.load_state_dict(state_dict)
    else:
        raise ValueError(f"{item} doesn't exists")

    cfg.merge_from_file(cfg_file)
    zcls_model = GhostNet(cfg)
    zcls_model.eval()

    zcls_model_dict = convert(official_model, zcls_model)
    zcls_model.load_state_dict(zcls_model_dict)

    res_dir = 'outputs/converters/'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    checkpoint = CheckPointer(model=zcls_model, save_dir=res_dir, save_to_disk=True)
    checkpoint.save(f'{item}_imagenet')


if __name__ == '__main__':
    item_list = ['ghostnet_x1_0', ]
    cfg_file_list = [
        'ghostnet_x1_0_zcls_imagenet_224.yaml',
    ]
    prefix_path = 'configs/benchmarks/ghostnet'
    for item, cfg_file in zip(item_list, cfg_file_list):
        config_path = os.path.join(prefix_path, cfg_file)
        print(item, config_path)
        process(item, config_path)

# -*- coding: utf-8 -*-

"""
@date: 2021/5/4 下午7:11
@file: torchvision_resnet_to_zcls_resnet.py
@author: zj
@description: Transform pretrained model into zcls format
official SKNet repos use caffe, so i found another repo
first, download SKNet repo and set env
git clone https://github.com/syt2/SKNet.git
export PYTHONPATH=$PYTHONPATH:/path/to/SKNet
then, download pretrained model
place the downloaded pretrained model in runs/sknet_imagenet/86028 folder under this project
"""

import os
import torch
from models.SKNet import SKNet
from utils import convert_state_dict

from zcls.model.backbones.resnet.resnet_backbone import arch_settings
from zcls.model.recognizers.build import build_recognizer
from zcls.util.checkpoint import CheckPointer
from zcls.config import cfg


def convert(official_model, zcls_model, stage_repeats):
    official_dict = official_model.state_dict()
    zcls_dict = zcls_model.state_dict()

    # k_list = [k for k, v in official_dict.items()]
    v_list = [v for k, v in official_dict.items()]

    idx = 6
    for num in stage_repeats:
        # k_list = k_list[:idx] + k_list[(idx + 34):(idx + 40)] + k_list[idx:(idx + 34)] + k_list[(idx + 40):]
        v_list = v_list[:idx] + v_list[(idx + 34):(idx + 40)] + v_list[idx:(idx + 34)] + v_list[(idx + 40):]
        # print(v_list[idx + 30].shape, v_list[idx + 32].shape)
        v_list[idx + 30] = v_list[idx + 30].squeeze()
        v_list[idx + 32] = v_list[idx + 32].squeeze()
        # print(v_list[idx + 30].shape, v_list[idx + 32].shape)

        idx += 40
        for i in range(num - 1):
            # print(v_list[idx + 24].shape, v_list[idx + 26].shape)
            v_list[idx + 24] = v_list[idx + 24].squeeze()
            v_list[idx + 26] = v_list[idx + 26].squeeze()
            # print(v_list[idx + 24].shape, v_list[idx + 26].shape)
            idx += 34

    for o_v, (z_k, z_v) in zip(v_list, zcls_dict.items()):
        zcls_dict[z_k] = o_v

    return zcls_dict


def process(item, cfg_file):
    if item == 'sknet50':
        official_model = SKNet()
        checkpoint = torch.load('/home/zj/repos/SKNet/runs/sknet_imagenet/86028/best_model.pkl', map_location='cpu')
        print(checkpoint.keys())
        official_model.load_state_dict(convert_state_dict(checkpoint["state_dict"]))
    else:
        raise ValueError(f"{item} doesn't exists")

    cfg.merge_from_file(cfg_file)
    zcls_model = build_recognizer(cfg, torch.device('cpu'))

    zcls_model_dict = convert(official_model, zcls_model, arch_settings[item][1])
    zcls_model.load_state_dict(zcls_model_dict)

    res_dir = 'outputs/converters/'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    checkpoint = CheckPointer(model=zcls_model, save_dir=res_dir, save_to_disk=True)
    checkpoint.save(f'{item}_imagenet')


if __name__ == '__main__':
    item_list = ['sknet50', ]
    cfg_file_list = [
        'sknet50_zcls_imagenet_224.yaml',
    ]
    prefix_path = 'configs/benchmarks/resnet-resnext'
    for item, cfg_file in zip(item_list, cfg_file_list):
        config_path = os.path.join(prefix_path, cfg_file)
        print(item, config_path)
        process(item, config_path)

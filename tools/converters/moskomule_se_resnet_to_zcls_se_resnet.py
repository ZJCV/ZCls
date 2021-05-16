# -*- coding: utf-8 -*-

"""
@date: 2021/5/4 下午7:11
@file: torchvision_resnet_to_zcls_resnet.py
@author: zj
@description: Transform official pretrained model into zcls format
official SENet repos use caffe, so i found another repo
use [moskomule/senet.pytorch](https://github.com/moskomule/senet.pytorch) pretrained model
"""

import os
import torch

from zcls.model.recognizers.build import build_recognizer
from zcls.util.checkpoint import CheckPointer
from zcls.config import get_cfg_defaults
from zcls.model.backbones.resnet.resnet_backbone import arch_settings


def convert(official_model, zcls_model, stage_repeats):
    official_dict = official_model.state_dict()
    zcls_dict = zcls_model.state_dict()

    k_list = [k for k, v in official_dict.items()]
    v_list = [v for k, v in official_dict.items()]

    idx = 6
    for num in stage_repeats:
        k_list = k_list[:idx] + k_list[(idx + 20):(idx + 26)] + k_list[idx:(idx + 20)] + k_list[(idx + 26):]
        v_list = v_list[:idx] + v_list[(idx + 20):(idx + 26)] + v_list[idx:(idx + 20)] + v_list[(idx + 26):]
        idx += 26
        idx += 20 * (num - 1)

    for o_k, o_v, (z_k, z_v) in zip(k_list, v_list, zcls_dict.items()):
        # print(o_k, o_v.shape, z_k, z_v.shape)
        zcls_dict[z_k] = o_v

    return zcls_dict


def process(item, cfg_file):
    if item == 'resnet50':
        official_model = torch.hub.load('moskomule/senet.pytorch', 'se_resnet50', pretrained=True)
    else:
        raise ValueError(f"{item} doesn't exists")

    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file)
    zcls_model = build_recognizer(cfg, torch.device('cpu'))

    zcls_model_dict = convert(official_model, zcls_model, arch_settings[item][1])
    zcls_model.load_state_dict(zcls_model_dict)

    res_dir = 'outputs/converters/'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    checkpoint = CheckPointer(model=zcls_model, save_dir=res_dir, save_to_disk=True)
    checkpoint.save(f'se_{item}_imagenet')


if __name__ == '__main__':
    item_list = ['resnet50']
    cfg_file_list = [
        'se_r50_zcls_imagenet_224.yaml',
    ]
    prefix_path = 'configs/benchmarks/resnet'
    for item, cfg_file in zip(item_list, cfg_file_list):
        config_path = os.path.join(prefix_path, cfg_file)
        print(item, config_path)
        process(item, config_path)

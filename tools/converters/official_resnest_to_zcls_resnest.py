# -*- coding: utf-8 -*-

"""
@date: 2021/5/4 下午7:11
@file: torchvision_resnet_to_zcls_resnet.py
@author: zj
@description: Transform official pretrained model into zcls format
"""

import os
import torch

from zcls.model.recognizers.build import build_recognizer
from zcls.util.checkpoint import CheckPointer
from zcls.config import get_cfg_defaults
from zcls.model.backbones.resnet.resnet_d_backbone import arch_settings


def convert(official_model, zcls_model, stage_repeats):
    official_dict = official_model.state_dict()
    zcls_dict = zcls_model.state_dict()

    k_list = [k for k, v in official_dict.items()]
    v_list = [v for k, v in official_dict.items()]

    idx = 18
    for num in stage_repeats:
        k_list = k_list[:idx] + k_list[(idx + 27):(idx + 33)] + k_list[idx:(idx + 27)] + k_list[(idx + 33):]
        v_list = v_list[:idx] + v_list[(idx + 27):(idx + 33)] + v_list[idx:(idx + 27)] + v_list[(idx + 33):]
        idx += 33
        idx += 27 * (num - 1)

    for o_k, o_v, (z_k, z_v) in zip(k_list, v_list, zcls_dict.items()):
        # print(o_k, o_v.shape, z_k, z_v.shape)
        zcls_dict[z_k] = o_v

    return zcls_dict


def process(item, cfg_file):
    if item == 'resnest50_fast_2s1x64d':
        official_model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50_fast_2s1x64d', pretrained=True)
    elif item == 'resnest50':
        official_model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
    elif item == 'resnest101':
        official_model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest101', pretrained=True)
    elif item == 'resnest200':
        official_model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest200', pretrained=True)
    elif item == 'resnest269':
        official_model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest269', pretrained=True)
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
    checkpoint.save(f'{item}_imagenet')


if __name__ == '__main__':
    item_list = ['resnest50_fast_2s1x64d', 'resnest50', 'resnest101', 'resnest200', 'resnest269']
    cfg_file_list = [
        'resnest50_fast_2s1x64d_zcls_imagenet_224.yaml',
        'resnest50_zcls_imagenet_224.yaml',
        'resnest101_zcls_imagenet_224.yaml',
        'resnest200_zcls_imagenet_224.yaml',
        'resnest269_zcls_imagenet_224.yaml',
    ]
    prefix_path = 'configs/benchmarks/resnet'
    for item, cfg_file in zip(item_list, cfg_file_list):
        config_path = os.path.join(prefix_path, cfg_file)
        print(item, config_path)
        process(item, config_path)

# -*- coding: utf-8 -*-

"""
@date: 2021/5/4 下午7:11
@file: torchvision_resnet_to_zcls_resnet.py
@author: zj
@description: Transform torchvision pretrained model into zcls format
"""

import os
from torchvision.models import mobilenet_v2, mnasnet0_5, mnasnet1_0
from zcls.model.recognizers.mobilenet.mobilenetv2 import MobileNetV2
from zcls.model.recognizers.mobilenet.mnasnet import MNASNet
from zcls.util.checkpoint import CheckPointer
from zcls.config import cfg


def convert_mobilenet_v2(torchvision_model, zcls_model):
    torchvision_resnet_dict = torchvision_model.state_dict()
    zcls_resnet_dict = zcls_model.state_dict()

    for (t_k, t_v), (z_k, z_v) in zip(torchvision_resnet_dict.items(), zcls_resnet_dict.items()):
        zcls_resnet_dict[z_k] = t_v

    return zcls_resnet_dict


def convert_mnasnet(torchvision_model, zcls_model):
    torchvision_resnet_dict = torchvision_model.state_dict()
    zcls_resnet_dict = zcls_model.state_dict()

    v_list = [v for k, v in torchvision_resnet_dict.items()]
    v_list = v_list[:6] + v_list[-8:-2] + v_list[6:-8] + v_list[-2:]

    for t_v, (z_k, z_v) in zip(v_list, zcls_resnet_dict.items()):
        zcls_resnet_dict[z_k] = t_v

    return zcls_resnet_dict


def process(item, cfg_file):
    if item == 'mobilenet_v2':
        torchvision_model = mobilenet_v2(pretrained=True)
    elif item == 'mnasnet0_5':
        torchvision_model = mnasnet0_5(pretrained=True)
    elif item == 'mnasnet1_0':
        torchvision_model = mnasnet1_0(pretrained=True)
    else:
        raise ValueError(f"{item} doesn't exists")

    cfg.merge_from_file(cfg_file)
    if item == 'mobilenet_v2':
        zcls_model = MobileNetV2(cfg)
        zcls_model_dict = convert_mobilenet_v2(torchvision_model, zcls_model)
        zcls_model.load_state_dict(zcls_model_dict)
    elif 'mnasnet' in item:
        zcls_model = MNASNet(cfg)
        zcls_model_dict = convert_mnasnet(torchvision_model, zcls_model)
        zcls_model.load_state_dict(zcls_model_dict)
    else:
        raise ValueError(f"{item} doesn't exists")

    res_dir = 'outputs/converters/'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    checkpoint = CheckPointer(model=zcls_model, save_dir=res_dir, save_to_disk=True)
    checkpoint.save(f'{item}_imagenet')


if __name__ == '__main__':
    item_list = ['mobilenet_v2', 'mnasnet0_5', 'mnasnet1_0']
    cfg_file_list = [
        'mobilenet_v2_zcls_imagenet_224.yaml',
        'mnasnet_b1_0_5_zcls_imagenet_224.yaml',
        'mnasnet_b1_1_0_zcls_imagenet_224.yaml',
    ]
    prefix_path = 'configs/benchmarks/lightweight'
    for item, cfg_file in zip(item_list, cfg_file_list):
        config_path = os.path.join(prefix_path, cfg_file)
        print(item, config_path)
        process(item, config_path)

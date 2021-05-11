# -*- coding: utf-8 -*-

"""
@date: 2021/5/4 下午7:11
@file: torchvision_resnet_to_zcls_resnet.py
@author: zj
@description: Transform torchvision pretrained model into zcls format
"""

import os
from torchvision.models import shufflenet_v2_x0_5, shufflenet_v2_x1_0
from zcls.model.recognizers.shufflenet.shufflenetv2 import ShuffleNetV2
from zcls.model.backbones.shufflenet.shufflenetv2_backbone import arch_settings
from zcls.util.checkpoint import CheckPointer
from zcls.config import cfg


def convert_shufflenet(torchvision_model, zcls_model, stage_repeats):
    torchvision_resnet_dict = torchvision_model.state_dict()
    zcls_resnet_dict = zcls_model.state_dict()

    v_list = [v for k, v in torchvision_resnet_dict.items()]
    v_list = v_list[:6] + v_list[-8:-2] + v_list[6:-8] + v_list[-2:]

    idx = 12
    for num in stage_repeats:
        v_list = v_list[:idx] + v_list[(idx + 12):(idx + 30)] + v_list[idx:(idx + 12)] + v_list[(idx + 30):]
        idx += 30
        idx += 18 * (num - 1)

    for t_v, (z_k, z_v) in zip(v_list, zcls_resnet_dict.items()):
        zcls_resnet_dict[z_k] = t_v

    return zcls_resnet_dict


def process(item, cfg_file):
    if item == 'shufflenet_v2_x0_5':
        torchvision_model = shufflenet_v2_x0_5(pretrained=True)
    elif item == 'shufflenet_v2_x1_0':
        torchvision_model = shufflenet_v2_x1_0(pretrained=True)
    else:
        raise ValueError(f"{item} doesn't exists")

    cfg.merge_from_file(cfg_file)
    zcls_model = ShuffleNetV2(cfg)
    zcls_model_dict = convert_shufflenet(torchvision_model, zcls_model, arch_settings[item][2])
    zcls_model.load_state_dict(zcls_model_dict)

    res_dir = 'outputs/converters/'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    checkpoint = CheckPointer(model=zcls_model, save_dir=res_dir, save_to_disk=True)
    checkpoint.save(f'{item}_imagenet')


if __name__ == '__main__':
    item_list = ['shufflenet_v2_x0_5', 'shufflenet_v2_x1_0']
    cfg_file_list = [
        'shufflenet_v2_x0_5_zcls_imagenet_224.yaml',
        'shufflenet_v2_x1_0_zcls_imagenet_224.yaml',
    ]
    prefix_path = 'configs/benchmarks/lightweight'
    for item, cfg_file in zip(item_list, cfg_file_list):
        config_path = os.path.join(prefix_path, cfg_file)
        print(item, config_path)
        process(item, config_path)

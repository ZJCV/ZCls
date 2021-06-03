# -*- coding: utf-8 -*-

"""
@date: 2021/5/4 下午7:11
@file: torchvision_resnet_to_zcls_resnet.py
@author: zj
@description: Transform official pretrained model into zcls format
refer to [megvii-model/ShuffleNet-Series](https://github.com/megvii-model/ShuffleNet-Series)
```
git clone https://github.com/megvii-model/ShuffleNet-Series.git
```

for shufflenetv1
1. set env
```
$ export PYTHONPATH=/path/to/ShuffleNet-Series/ShuffleNetV1
```
2. download pretrained model
"""

import os
import torch
from network import ShuffleNetV1
from zcls.model.recognizers.build import build_recognizer
from zcls.util.checkpoint import CheckPointer
from zcls.config import get_cfg_defaults


def convert(official_dict, zcls_model):
    zcls_resnet_dict = zcls_model.state_dict()

    for (t_k, t_v), (z_k, z_v) in zip(official_dict.items(), zcls_resnet_dict.items()):
        zcls_resnet_dict[z_k] = t_v

    return zcls_resnet_dict


def process(item, cfg_file):
    if item == 'shufflenetv1_3g2x':
        official_dict = torch.load('outputs/ShuffleNetV1/Group3/models/2.0x.pth.tar', torch.device('cpu'))['state_dict']
    elif item == 'shufflenetv1_3g1_5x':
        official_dict = torch.load('outputs/ShuffleNetV1/Group3/models/1.5x.pth.tar', torch.device('cpu'))['state_dict']
    elif item == 'shufflenetv1_3g1x':
        official_dict = torch.load('outputs/ShuffleNetV1/Group3/models/1.0x.pth.tar', torch.device('cpu'))['state_dict']
    elif item == 'shufflenetv1_3g0_5x':
        official_dict = torch.load('outputs/ShuffleNetV1/Group3/models/0.5x.pth.tar', torch.device('cpu'))['state_dict']
    elif item == 'shufflenetv1_8g2x':
        official_dict = \
        torch.load('outputs/ShuffleNetV1/Group8/models/snetv1_group8_2.0x.pth.tar', torch.device('cpu'))['state_dict']
    elif item == 'shufflenetv1_8g1_5x':
        official_dict = \
        torch.load('outputs/ShuffleNetV1/Group8/models/snetv1_group8_1.5x.pth.tar', torch.device('cpu'))['state_dict']
    elif item == 'shufflenetv1_8g1x':
        official_dict = \
        torch.load('outputs/ShuffleNetV1/Group8/models/snetv1_group8_1.0x.pth.tar', torch.device('cpu'))['state_dict']
    elif item == 'shufflenetv1_8g0_5x':
        official_dict = \
        torch.load('outputs/ShuffleNetV1/Group8/models/snetv1_group8_0.5x.pth.tar', torch.device('cpu'))['state_dict']
    else:
        raise ValueError(f"{item} doesn't exists")

    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file)
    zcls_model = build_recognizer(cfg, torch.device('cpu'))

    zcls_model_dict = convert(official_dict, zcls_model)
    zcls_model.load_state_dict(zcls_model_dict)

    res_dir = 'outputs/converters/'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    checkpoint = CheckPointer(model=zcls_model, save_dir=res_dir, save_to_disk=True)
    checkpoint.save(f'{item}_imagenet')


if __name__ == '__main__':
    item_list = ['shufflenetv1_3g2x', 'shufflenetv1_3g1_5x', 'shufflenetv1_3g1x', 'shufflenetv1_3g0_5x',
                 'shufflenetv1_8g2x', 'shufflenetv1_8g1_5x', 'shufflenetv1_8g1x', 'shufflenetv1_8g0_5x']
    cfg_file_list = [
        'shufflenet_v1_3g2x_zcls_imagenet_224.yaml',
        'shufflenet_v1_3g1_5x_zcls_imagenet_224.yaml',
        'shufflenet_v1_3g1x_zcls_imagenet_224.yaml',
        'shufflenet_v1_3g0_5x_zcls_imagenet_224.yaml',
        'shufflenet_v1_8g2x_zcls_imagenet_224.yaml',
        'shufflenet_v1_8g1_5x_zcls_imagenet_224.yaml',
        'shufflenet_v1_8g1x_zcls_imagenet_224.yaml',
        'shufflenet_v1_8g0_5x_zcls_imagenet_224.yaml',
    ]
    prefix_path = 'configs/benchmarks/shufflenet'
    for item, cfg_file in zip(item_list, cfg_file_list):
        config_path = os.path.join(prefix_path, cfg_file)
        print(item, config_path)
        process(item, config_path)

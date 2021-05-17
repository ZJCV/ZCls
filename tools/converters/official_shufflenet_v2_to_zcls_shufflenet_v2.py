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

for shufflenetv2
1. set env
```
$ export PYTHONPATH=/path/to/ShuffleNet-Series/ShuffleNetV2
```
2. download pretrained model
"""

import os
import torch
from network import ShuffleNetV2
from zcls.model.recognizers.build import build_recognizer
from zcls.util.checkpoint import CheckPointer
from zcls.config import get_cfg_defaults


def convert(official_dict, zcls_model):
    zcls_resnet_dict = zcls_model.state_dict()

    o_v_list = [v for k, v in official_dict.items()]
    o_v_list = o_v_list[:6] + o_v_list[330:336] + o_v_list[6:330] + o_v_list[336:]

    for o_v, (z_k, z_v) in zip(o_v_list, zcls_resnet_dict.items()):
        zcls_resnet_dict[z_k] = o_v

    return zcls_resnet_dict


def process(item, cfg_file):
    if item == 'shufflenet_v2_x0_5':
        official_dict = torch.load('outputs/ShufflenetV2/models/ShuffleNetV2.0.5x.pth.tar', torch.device('cpu'))['state_dict']
    elif item == 'shufflenet_v2_x1_0':
        official_dict = torch.load('outputs/ShufflenetV2/models/ShuffleNetV2.1.0x.pth.tar', torch.device('cpu'))['state_dict']
    elif item == 'shufflenet_v2_x1_5':
        official_dict = torch.load('outputs/ShufflenetV2/models/ShuffleNetV2.1.5x.pth.tar', torch.device('cpu'))['state_dict']
    elif item == 'shufflenet_v2_x2_0':
        official_dict = torch.load('outputs/ShufflenetV2/models/ShuffleNetV2.2.0x.pth.tar', torch.device('cpu'))['state_dict']
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
    item_list = ['shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0']
    cfg_file_list = [
        'shufflenet_v2_x0_5_zcls_imagenet_224.yaml',
        'shufflenet_v2_x1_0_zcls_imagenet_224.yaml',
        'shufflenet_v2_x1_5_zcls_imagenet_224.yaml',
        'shufflenet_v2_x2_0_zcls_imagenet_224.yaml'
    ]
    prefix_path = 'configs/benchmarks/shufflenet'
    for item, cfg_file in zip(item_list, cfg_file_list):
        config_path = os.path.join(prefix_path, cfg_file)
        print(item, config_path)
        process(item, config_path)

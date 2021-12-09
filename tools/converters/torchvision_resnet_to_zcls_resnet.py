# -*- coding: utf-8 -*-

"""
@date: 2021/5/4 下午7:11
@file: torchvision_resnet_to_zcls_resnet.py
@author: zj
@description: Transform torchvision pretrained model into zcls format
"""

import os
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, \
    resnext101_32x8d
from zcls.model.recognizers.resnet.resnet import ResNet
from zcls.config import cfg
from zcls.util.checkpoint import CheckPointer


def convert(torchvision_resnet, zcls_resnet):
    torchvision_resnet_dict = torchvision_resnet.state_dict()
    zcls_resnet_dict = zcls_resnet.state_dict()

    for k, v in torchvision_resnet_dict.items():
        if 'downsample' in k:
            zcls_resnet_dict[f"backbone.{k.replace('downsample', 'down_sample')}"] = v
        elif 'layer' in k:
            zcls_resnet_dict[f'backbone.{k}'] = v
        elif 'fc' in k:
            zcls_resnet_dict[f'head.{k}'] = v
        elif 'conv1.weight' == k:
            zcls_resnet_dict['backbone.stem.0.weight'] = v
        elif 'bn1' in k:
            zcls_resnet_dict[k.replace('bn1', 'backbone.stem.1')] = v
        else:
            raise ValueError("{k} doesn't exist")

    return zcls_resnet_dict


def process(item, cfg_file):
    if item == 'resnet18':
        torchvision_resnet = resnet18(pretrained=True)
    elif item == 'resnet34':
        torchvision_resnet = resnet34(pretrained=True)
    elif item == 'resnet50':
        torchvision_resnet = resnet50(pretrained=True)
    elif item == 'resnet101':
        torchvision_resnet = resnet101(pretrained=True)
    elif item == 'resnet152':
        torchvision_resnet = resnet152(pretrained=True)
    elif item == 'resnext50_32x4d':
        torchvision_resnet = resnext50_32x4d(pretrained=True)
    elif item == 'resnext101_32x8d':
        torchvision_resnet = resnext101_32x8d(pretrained=True)
    else:
        raise ValueError(f"{item} doesn't exists")

    cfg.merge_from_file(cfg_file)
    zcls_resnet = ResNet(cfg)

    zcls_resnet_dict = convert(torchvision_resnet, zcls_resnet)
    zcls_resnet.load_state_dict(zcls_resnet_dict)

    res_dir = 'outputs/converters/'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    checkpoint = CheckPointer(model=zcls_resnet, save_dir=res_dir, save_to_disk=True)
    checkpoint.save(f'{item}_imagenet')


if __name__ == '__main__':
    item_list = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']
    cfg_file_list = [
        'r18_zcls_imagenet_224.yaml',
        'r34_zcls_imagenet_224.yaml',
        'r50_zcls_imagenet_224.yaml',
        'r101_zcls_imagenet_224.yaml',
        'r152_zcls_imagenet_224.yaml',
        'rxt50_32x4d_zcls_imagenet_224.yaml',
        'rxt101_32x8d_zcls_imagenet_224.yaml'
    ]
    prefix_path = 'configs/benchmarks/resnet-resnext'
    for item, cfg_file in zip(item_list, cfg_file_list):
        config_path = os.path.join(prefix_path, cfg_file)
        print(config_path)
        process(item, config_path)

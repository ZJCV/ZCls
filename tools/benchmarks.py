# -*- coding: utf-8 -*-

"""
@date: 2020/11/4 下午2:06
@file: benchmarks.py
@author: zj
@description: 
"""

import time
import numpy as np
import torch

from zcls.util.metrics import compute_num_flops
from zcls.config import get_cfg_defaults
from zcls.model.recognizers.build import build_recognizer


def compute_model_time(data_shape, model, device):
    model = model.to(device)

    t1 = 0.0
    num = 100
    begin = time.time()
    for i in range(num):
        data = torch.randn(data_shape)
        start = time.time()
        model(data.to(device=device, non_blocking=True))
        if i > num // 2:
            t1 += time.time() - start
    t2 = time.time() - begin
    print(f'one process need {t2 / num:.3f}s, model compute need: {t1 / (num // 2):.3f}s')


def main(data_shape, config_file, mobile_name):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_file)

    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # gpu_device = torch.device('cuda:0')
    cpu_device = torch.device('cpu')

    model = build_recognizer(cfg, cpu_device)
    model.eval()
    data = torch.randn(data_shape).to(device=cpu_device, non_blocking=True)

    GFlops, params_size = compute_num_flops(model, data)
    print(f'{mobile_name} ' + '*' * 10)
    print(f'device: {cpu_device}')
    print(f'GFlops: {GFlops:.3f}G')
    print(f'Params Size: {params_size:.3f}MB')

    model = build_recognizer(cfg, cpu_device)
    model.eval()
    print(f'compute cpu infer time')
    compute_model_time(data_shape, model, cpu_device)
    # print(f'compute gpu infer time')
    # compute_model_time(data_shape, model, gpu_device)

    del model
    torch.cuda.empty_cache()


def mobilenet():
    data_shape = (1, 3, 224, 224)

    cfg_file_list = [
        'configs/benchmarks/mobilenet/mnasnet_b1_0_5_torchvision_imagenet_224.yaml',
        'configs/benchmarks/mobilenet/mnasnet_b1_0_5_zcls_imagenet_224.yaml',
        'configs/benchmarks/mobilenet/mnasnet_b1_1_0_torchvision_imagenet_224.yaml',
        'configs/benchmarks/mobilenet/mnasnet_b1_1_0_zcls_imagenet_224.yaml',
        'configs/benchmarks/mobilenet/mobilenet_v2_torchvision_imagenet_224.yaml',
        'configs/benchmarks/mobilenet/mobilenet_v2_zcls_imagenet_224.yaml',
    ]

    name_list = [
        'mnasnet_b1_0_5_torchvision',
        'mnasnet_b1_0_5_zcls',
        'mnasnet_b1_1_0_torchvision',
        'mnasnet_b1_1_0_zcls',
        'mobilenet_v2_torchvision',
        'mobilenet_v2_zcls',
    ]

    assert len(name_list) == len(cfg_file_list)

    for name, cfg_file in zip(name_list, cfg_file_list):
        main(data_shape, cfg_file, name)


def shufflenet():
    data_shape = (1, 3, 224, 224)

    cfg_file_list = [
        'configs/benchmarks/shufflenet/shufflenet_v2_x0_5_torchvision_imagenet_224.yaml',
        'configs/benchmarks/shufflenet/shufflenet_v2_x0_5_zcls_imagenet_224.yaml',
        'configs/benchmarks/shufflenet/shufflenet_v2_x1_0_torchvision_imagenet_224.yaml',
        'configs/benchmarks/shufflenet/shufflenet_v2_x1_0_zcls_imagenet_224.yaml',
        'configs/benchmarks/shufflenet/shufflenet_v1_3g0_5x_zcls_imagenet_224.yaml',
        'configs/benchmarks/shufflenet/shufflenet_v1_3g1_5x_zcls_imagenet_224.yaml',
        'configs/benchmarks/shufflenet/shufflenet_v1_3g1x_zcls_imagenet_224.yaml',
        'configs/benchmarks/shufflenet/shufflenet_v1_3g2x_zcls_imagenet_224.yaml',
        'configs/benchmarks/shufflenet/shufflenet_v1_8g0_5x_zcls_imagenet_224.yaml',
        'configs/benchmarks/shufflenet/shufflenet_v1_8g1_5x_zcls_imagenet_224.yaml',
        'configs/benchmarks/shufflenet/shufflenet_v1_8g1x_zcls_imagenet_224.yaml',
        'configs/benchmarks/shufflenet/shufflenet_v1_8g2x_zcls_imagenet_224.yaml',
    ]

    name_list = [
        'shufflenet_v2_x0_5_torchvision',
        'shufflenet_v2_x0_5_zcls',
        'shufflenet_v2_x1_0_torchvision',
        'shufflenet_v2_x1_0_zcls',
        'shufflenet_v1_3g0_5x_zcls',
        'shufflenet_v1_3g1_5x_zcls',
        'shufflenet_v1_3g1x_zcls',
        'shufflenet_v1_3g2x_zcls',
        'shufflenet_v1_8g0_5x_zcls',
        'shufflenet_v1_8g1_5x_zcls',
        'shufflenet_v1_8g1x_zcls',
        'shufflenet_v1_8g2x_zcls',
    ]

    assert len(name_list) == len(cfg_file_list)

    for name, cfg_file in zip(name_list, cfg_file_list):
        main(data_shape, cfg_file, name)


def resnet():
    data_shape = (1, 3, 224, 224)

    cfg_file_list = [
        'configs/benchmarks/resnet/r18_torchvision_imagenet_224.yaml',
        'configs/benchmarks/resnet/r18_zcls_imagenet_224.yaml',
        'configs/benchmarks/resnet/r34_torchvision_imagenet_224.yaml',
        'configs/benchmarks/resnet/r34_zcls_imagenet_224.yaml',
        'configs/benchmarks/resnet/r50_torchvision_imagenet_224.yaml',
        'configs/benchmarks/resnet/r50_zcls_imagenet_224.yaml',
        'configs/benchmarks/resnet/r101_torchvision_imagenet_224.yaml',
        'configs/benchmarks/resnet/r101_zcls_imagenet_224.yaml',
        'configs/benchmarks/resnet/r152_torchvision_imagenet_224.yaml',
        'configs/benchmarks/resnet/r152_zcls_imagenet_224.yaml',
        'configs/benchmarks/resnet/rxt50_32x4d_torchvision_imagenet_224.yaml',
        'configs/benchmarks/resnet/rxt50_32x4d_zcls_imagenet_224.yaml',
        'configs/benchmarks/resnet/rxt101_32x8d_torchvision_imagenet_224.yaml',
        'configs/benchmarks/resnet/rxt101_32x8d_zcls_imagenet_224.yaml',
        'configs/benchmarks/resnet/sknet50_zcls_imagenet_224.yaml'
    ]

    name_list = [
        'r18_torchvision',
        'r18_zcls',
        'r34_torchvision',
        'r34_zcls',
        'r50_torchvision',
        'r50_zcls',
        'r101_torchvision',
        'r101_zcls',
        'r152_torchvision',
        'r152_zcls',
        'rxt50_32x4d_torchvision',
        'rxt50_32x4d_zcls',
        'rxt101_32x8d_torchvision',
        'rxt101_32x8d_zcls',
        'sknet50_zcls_imagenet',
    ]

    assert len(name_list) == len(cfg_file_list)

    for name, cfg_file in zip(name_list, cfg_file_list):
        main(data_shape, cfg_file, name)


def repvgg():
    cfg_file_list = [
        'configs/benchmarks/repvgg/repvgg_a0_infer_zcls_imagenet_224.yaml',
        'configs/benchmarks/repvgg/repvgg_a0_train_zcls_imagenet_224.yaml',
        'configs/benchmarks/repvgg/repvgg_a1_infer_zcls_imagenet_224.yaml',
        'configs/benchmarks/repvgg/repvgg_a1_train_zcls_imagenet_224.yaml',
        'configs/benchmarks/repvgg/repvgg_a2_infer_zcls_imagenet_224.yaml',
        'configs/benchmarks/repvgg/repvgg_a2_train_zcls_imagenet_224.yaml',
        'configs/benchmarks/repvgg/repvgg_b0_infer_zcls_imagenet_224.yaml',
        'configs/benchmarks/repvgg/repvgg_b0_train_zcls_imagenet_224.yaml',
        'configs/benchmarks/repvgg/repvgg_b1_infer_zcls_imagenet_224.yaml',
        'configs/benchmarks/repvgg/repvgg_b1_train_zcls_imagenet_224.yaml',
        'configs/benchmarks/repvgg/repvgg_b1g2_infer_zcls_imagenet_224.yaml',
        'configs/benchmarks/repvgg/repvgg_b1g2_train_zcls_imagenet_224.yaml',
        'configs/benchmarks/repvgg/repvgg_b1g4_infer_zcls_imagenet_224.yaml',
        'configs/benchmarks/repvgg/repvgg_b1g4_train_zcls_imagenet_224.yaml',
        'configs/benchmarks/repvgg/repvgg_b2_infer_zcls_imagenet_224.yaml',
        'configs/benchmarks/repvgg/repvgg_b2_train_zcls_imagenet_224.yaml',
        'configs/benchmarks/repvgg/repvgg_b2g4_infer_zcls_imagenet_224.yaml',
        'configs/benchmarks/repvgg/repvgg_b2g4_train_zcls_imagenet_224.yaml',
        'configs/benchmarks/repvgg/repvgg_b3_infer_zcls_imagenet_224.yaml',
        'configs/benchmarks/repvgg/repvgg_b3_train_zcls_imagenet_224.yaml',
        'configs/benchmarks/repvgg/repvgg_b3g4_infer_zcls_imagenet_224.yaml',
        'configs/benchmarks/repvgg/repvgg_b3g4_infer_zcls_imagenet_320.yaml',
        'configs/benchmarks/repvgg/repvgg_b3g4_train_zcls_imagenet_224.yaml',
        'configs/benchmarks/repvgg/repvgg_b3g4_train_zcls_imagenet_320.yaml',
        'configs/benchmarks/repvgg/repvgg_d2se_infer_zcls_imagenet_224.yaml',
        'configs/benchmarks/repvgg/repvgg_d2se_infer_zcls_imagenet_320.yaml',
        'configs/benchmarks/repvgg/repvgg_d2se_train_zcls_imagenet_224.yaml',
        'configs/benchmarks/repvgg/repvgg_d2se_train_zcls_imagenet_320.yaml',
    ]

    name_list = [
        'repvgg_a0_infer_zcls',
        'repvgg_a0_train_zcls',
        'repvgg_a1_infer_zcls',
        'repvgg_a1_train_zcls',
        'repvgg_a2_infer_zcls',
        'repvgg_a2_train_zcls',
        'repvgg_b0_infer_zcls',
        'repvgg_b0_train_zcls',
        'repvgg_b1_infer_zcls',
        'repvgg_b1_train_zcls',
        'repvgg_b1g2_infer_zcls',
        'repvgg_b1g2_train_zcls',
        'repvgg_b1g4_infer_zcls',
        'repvgg_b1g4_train_zcls',
        'repvgg_b2_infer_zcls',
        'repvgg_b2_train_zcls',
        'repvgg_b2g4_infer_zcls',
        'repvgg_b2g4_train_zcls',
        'repvgg_b3_infer_zcls',
        'repvgg_b3_train_zcls',
        'repvgg_b3g4_infer_zcls_224',
        'repvgg_b3g4_infer_zcls_320',
        'repvgg_b3g4_train_zcls_224',
        'repvgg_b3g4_train_zcls_320',
        'repvgg_d2se_infer_zcls_224',
        'repvgg_d2se_infer_zcls_320',
        'repvgg_d2se_train_zcls_224',
        'repvgg_d2se_train_zcls_320',
    ]

    # print(len(name_list), len(cfg_file_list))
    assert len(name_list) == len(cfg_file_list)

    for name, cfg_file in zip(name_list, cfg_file_list):
        if '224' in cfg_file:
            data_shape = (1, 3, 224, 224)
            main(data_shape, cfg_file, name)
        elif '320' in cfg_file:
            data_shape = (1, 3, 320, 320)
            main(data_shape, cfg_file, name)
        else:
            raise ValueError('ERROR')


def resnest():
    data_shape = (1, 3, 224, 224)

    cfg_file_list = [
        'configs/benchmarks/resnet/resnest50_fast_2s1x64d_zcls_imagenet_224.yaml',
        'configs/benchmarks/resnet/resnest50_fast_2s1x64d_official_imagenet_224.yaml',
        'configs/benchmarks/resnet/resnest50_zcls_imagenet_224.yaml',
        'configs/benchmarks/resnet/resnest50_official_imagenet_224.yaml',
        'configs/benchmarks/resnet/resnest101_zcls_imagenet_224.yaml',
        'configs/benchmarks/resnet/resnest101_official_imagenet_224.yaml',
        'configs/benchmarks/resnet/resnest200_zcls_imagenet_224.yaml',
        'configs/benchmarks/resnet/resnest200_official_imagenet_224.yaml',
        'configs/benchmarks/resnet/resnest269_zcls_imagenet_224.yaml',
        'configs/benchmarks/resnet/resnest269_official_imagenet_224.yaml',
    ]

    name_list = [
        'resnest50_fast_2s1x64d_zcls',
        'resnest50_fast_2s1x64d_official',
        'resnest50_zcls',
        'resnest50_official',
        'resnest101_zcls',
        'resnest101_official',
        'resnest200_zcls',
        'resnest200_official',
        'resnest269_zcls',
        'resnest269_official',
    ]

    assert len(name_list) == len(cfg_file_list)

    for name, cfg_file in zip(name_list, cfg_file_list):
        main(data_shape, cfg_file, name)


if __name__ == '__main__':
    # print('#' * 30)
    # mobilenet()
    print('#' * 30)
    shufflenet()
    # print('#' * 30)
    # resnet()
    # print('#' * 30)
    # repvgg()
    # print('#' * 30)
    # resnest()

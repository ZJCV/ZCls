# -*- coding: utf-8 -*-

"""
@date: 2020/11/4 下午2:06
@file: compute_flops.py
@author: zj
@description: 
"""

import time
import numpy as np
import torch

from zcls.util.distributed import get_device, get_local_rank
from zcls.util.metrics import compute_num_flops
from zcls.config import cfg
from zcls.model.recognizers.build import build_recognizer


def main(data_shape, config_file, mobile_name):
    cfg.merge_from_file(config_file)

    device = get_device(local_rank=get_local_rank())
    model = build_recognizer(cfg, device)
    model.eval()
    data = torch.randn(data_shape).to(device=device, non_blocking=True)

    GFlops, params_size = compute_num_flops(model, data)
    print(f'{mobile_name} ' + '*' * 10)
    print(f'device: {device}')
    print(f'GFlops: {GFlops:.3f}G')
    print(f'Params Size: {params_size:.3f}MB')

    data = torch.randn(data_shape)
    t1 = 0.0
    num = 100
    begin = time.time()
    for i in range(num):
        start = time.time()
        model(data.to(device=device, non_blocking=True))
        t1 += time.time() - start
    t2 = time.time() - begin
    print(f'one process need {t2 / num:.3f}s, model compute need: {t1 / num:.3f}s')


if __name__ == '__main__':
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    data_shape = (1, 3, 224, 224)

    # cfg_file = 'configs/resnet/r50_custom_cifar100_224.yaml'
    # name = 'ResNet_Custom'
    # main(data_shape, cfg_file, name)

    cfg_file = 'configs/mobilenet/mbv1_0.25x_cifar100_224.yaml'
    name = 'MobileNetV1_0.25x'
    main(data_shape, cfg_file, name)

    cfg_file = 'configs/mobilenet/mbv1_0.5x_cifar100_224.yaml'
    name = 'MobileNetV1_0.5x'
    main(data_shape, cfg_file, name)

    cfg_file = 'configs/mobilenet/mbv1_0.75x_cifar100_224.yaml'
    name = 'MobileNetV1_0.75x'
    main(data_shape, cfg_file, name)

    cfg_file = 'configs/mobilenet/mbv1_1x_cifar100_224.yaml'
    name = 'MobileNetV1_1x'
    main(data_shape, cfg_file, name)

    cfg_file = 'configs/mobilenet/mbv2_custom_1x_relu6_cifar100_224.yaml'
    name = 'MobileNetV2_custom_1x_relu6'
    main(data_shape, cfg_file, name)

    cfg_file = 'configs/mobilenet/mbv2_pytorch_1x_relu6_cifar100_224.yaml'
    name = 'MobileNetV2_pytorch_1x_relu6'
    main(data_shape, cfg_file, name)
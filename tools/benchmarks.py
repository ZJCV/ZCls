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

from zcls.util.distributed import get_device, get_local_rank
from zcls.util.metrics import compute_num_flops
from zcls.config import cfg
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
        t1 += time.time() - start
    t2 = time.time() - begin
    print(f'one process need {t2 / num:.3f}s, model compute need: {t1 / num:.3f}s')


def main(data_shape, config_file, mobile_name):
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    cfg.merge_from_file(config_file)

    gpu_device = torch.device('cuda:0')
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
    print(f'compute gpu infer time')
    compute_model_time(data_shape, model, gpu_device)

    del model
    torch.cuda.empty_cache()


def mobilenet():
    data_shape = (1, 3, 224, 224)

    cfg_file = 'configs/benchmarks/lightweight/mbv1_custom_cifar100_224_e100.yaml'
    name = 'MobileNetV1_1.0x'
    main(data_shape, cfg_file, name)

    cfg_file = 'configs/benchmarks/lightweight/mbv2_custom_cifar100_224_e100.yaml'
    name = 'MobileNetV2_1.0x'
    main(data_shape, cfg_file, name)

    cfg_file = 'configs/benchmarks/lightweight/mbv2_torchvision_cifar100_224_e100.yaml'
    name = 'Torchvision_MobileNetV2_1.0x'
    main(data_shape, cfg_file, name)

    cfg_file = 'configs/benchmarks/lightweight/mnasnet_a1_1_3_custom_cifar100_224_e100.yaml'
    name = 'MNasNet_a1_1.3x'
    main(data_shape, cfg_file, name)

    cfg_file = 'configs/benchmarks/lightweight/mnasnet_a1_1_3_se_custom_cifar100_224_e100.yaml'
    name = 'MNasNet_SE_a1_1.3x'
    main(data_shape, cfg_file, name)

    cfg_file = 'configs/benchmarks/lightweight/mnasnet_b1_1_3_custom_cifar100_224_e100_sgd.yaml'
    name = 'MNasNet_b1_1.3x'
    main(data_shape, cfg_file, name)

    cfg_file = 'configs/benchmarks/lightweight/mnasnet_b1_1_3_torchvision_cifar100_224_e100_sgd.yaml'
    name = 'Torchvision_MNasNet_b1_1.3x'
    main(data_shape, cfg_file, name)

    cfg_file = 'configs/benchmarks/lightweight/mbv3_large_custom_cifar100_224_e100_sgd.yaml'
    name = 'MobileNetV3_Large_1.0x'
    main(data_shape, cfg_file, name)

    cfg_file = 'configs/benchmarks/lightweight/mbv3_large_se_custom_cifar100_224_e100_sgd.yaml'
    name = 'MobileNetV3_SE_Large_1.0x'
    main(data_shape, cfg_file, name)

    cfg_file = 'configs/benchmarks/lightweight/mbv3_large_se_hsigmoid_custom_cifar100_224_e100.yaml'
    name = 'MobileNetV3_SE_HSigmoid_Large_1.0x'
    main(data_shape, cfg_file, name)

    cfg_file = 'configs/benchmarks/lightweight/mbv3_small_custom_cifar100_224_e100_sgd.yaml'
    name = 'MobileNetV3_Small_1.0x'
    main(data_shape, cfg_file, name)

    cfg_file = 'configs/benchmarks/lightweight/mbv3_small_se_custom_cifar100_224_e100.yaml'
    name = 'MobileNetV3_SE_Small_1.0x'
    main(data_shape, cfg_file, name)

    cfg_file = 'configs/benchmarks/lightweight/mbv3_small_se_hsigmoid_custom_cifar100_224_e100.yaml'
    name = 'MobileNetV3_SE_HSigmoid_Small_1.0x'
    main(data_shape, cfg_file, name)


def shufflenet():
    data_shape = (1, 3, 224, 224)

    cfg_file = 'configs/benchmarks/lightweight/sfv1_custom_cifar100_224_e100.yaml'
    name = 'ShuffleNetV1_1.0x'
    main(data_shape, cfg_file, name)

    cfg_file = 'configs/benchmarks/lightweight/sfv2_custom_cifar100_224_e100.yaml'
    name = 'ShuffleNetV2_1.0x'
    main(data_shape, cfg_file, name)

    cfg_file = 'configs/benchmarks/lightweight/sfv2_torchvision_cifar100_224_e100.yaml'
    name = 'Torchvision_ShuffleNetV2_1.0x'
    main(data_shape, cfg_file, name)


def resnet():
    data_shape = (1, 3, 224, 224)

    cfg_file = 'configs/benchmarks/resnet/r50_custom_cifar100_224_e100_rmsprop.yaml'
    name = 'ResNet50'
    main(data_shape, cfg_file, name)

    cfg_file = 'configs/benchmarks/resnet/r50_torchvision_cifar100_224_e100_rmsprop.yaml'
    name = 'Torchvision_ResNet50'
    main(data_shape, cfg_file, name)

    cfg_file = 'configs/benchmarks/resnet/rd50_custom_cifar100_224_e100_rmsprop.yaml'
    name = 'ResNetD50'
    main(data_shape, cfg_file, name)

    cfg_file = 'configs/benchmarks/resnet/rd50_custom_cifar100_224_e100_sgd.yaml'
    name = 'ResNetD50'
    main(data_shape, cfg_file, name)

    cfg_file = 'configs/benchmarks/resnet/rxd50_32x4d_avg_custom_cifar100_224_e100_rmsprop.yaml'
    name = 'ResNeXtD50_32x4d_avg'
    main(data_shape, cfg_file, name)

    cfg_file = 'configs/benchmarks/resnet/rxd50_32x4d_fast_avg_custom_cifar100_224_e100_rmsprop.yaml'
    name = 'ResNeXtD50_32x4d_fast_avg'
    main(data_shape, cfg_file, name)

    cfg_file = 'configs/benchmarks/resnet/rxt50_32x4d_custom_cifar100_224_e100_rmsprop.yaml'
    name = 'ResNeXt50_32x4d'
    main(data_shape, cfg_file, name)

    cfg_file = 'configs/benchmarks/resnet/rxt50_32x4d_custom_cifar100_224_e100_sgd.yaml'
    name = 'ResNeXt50_32x4d'
    main(data_shape, cfg_file, name)

    cfg_file = 'configs/benchmarks/resnet/rxt50_32x4d_torchvision_cifar100_224_e100_rmsprop.yaml'
    name = 'Torchvisoin_ResNeXt_32x4d'
    main(data_shape, cfg_file, name)

    cfg_file = 'configs/benchmarks/resnet/rxt50_32x4d_torchvision_cifar100_224_e100_sgd.yaml'
    name = 'Torchvision_ResNeXt50_32x4d'
    main(data_shape, cfg_file, name)

    cfg_file = 'configs/benchmarks/resnet/rxtd50_32x4d_custom_cifar100_224_e100_rmsprop.yaml'
    name = 'ResNeXtD50_32x4d'
    main(data_shape, cfg_file, name)

    cfg_file = 'configs/benchmarks/resnet/rxtd50_32x4d_custom_cifar100_224_e100_sgd.yaml'
    name = 'ResNeXtD50_32x4d'
    main(data_shape, cfg_file, name)

    cfg_file = 'configs/benchmarks/resnet/sknet50_custom_cifar100_224_e100_rmsprop.yaml'
    name = 'SKNet50'
    main(data_shape, cfg_file, name)

    cfg_file = 'configs/benchmarks/resnet/rst50_2s2x40d_custom_cifar100_224_e100_rmsprop.yaml'
    name = 'ResNeSt50_2s2x40d'
    main(data_shape, cfg_file, name)

    cfg_file = 'configs/benchmarks/resnet/rst50_2s2x40d_fast_custom_cifar100_224_e100_rmsprop.yaml'
    name = 'ResNeSt50_fast_2s2x40d'
    main(data_shape, cfg_file, name)

    cfg_file = 'configs/benchmarks/resnet/rst50_2s2x40d_official_cifar100_224_e100_rmsprop.yaml'
    name = 'Torchvision_ResNeSt50_2s2x40d'
    main(data_shape, cfg_file, name)

    cfg_file = 'configs/benchmarks/resnet/rst50_2s2x40d_fast_official_cifar100_224_e100_rmsprop.yaml'
    name = 'Torchvision_ResNeSt50_fast_2s2x40d'
    main(data_shape, cfg_file, name)


if __name__ == '__main__':
    # print('#' * 30)
    # mobilenet()
    # print('#' * 30)
    # shufflenet()
    print('#' * 30)
    resnet()

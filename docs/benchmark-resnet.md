
# Benchmark - ResNet

## Convert

The torchvision pretraining model can be transformed into zcls by script

```
$ cd /path/to/ZCls
$ python tools/converters/torchvision_resnet_to_zcls_resnet.py
```

## Benchmark

|       arch       |  framework  |  top1  |  top5  |
|:----------------:|:-----------:|:------:|:------:|
|     resnet18     |     zcls    | 69.224 | 88.808 |
|     resnet18     | torchvision | 69.222 | 88.808 |
|     resnet34     |     zcls    | 72.821 | 91.071 |
|     resnet34     | torchvision | 72.817 | 91.073 |
|     resnet50     |     zcls    | 75.692 | 92.768 |
|     resnet50     | torchvision | 75.690 | 92.770 |
|     resnet101    |     zcls    | 76.965 | 93.526 |
|     resnet101    | torchvision | 76.967 | 93.528 |
|     resnet152    |     zcls    | 78.141 | 93.946 |
|     resnet152    | torchvision | 78.139 | 93.946 |
|  resnext50_32x4d |     zcls    | 77.353 | 93.610 |
|  resnext50_32x4d | torchvision | 77.353 | 93.612 |
|  resnet101_32x8d |     zcls    | 79.075 | 94.540 |
| resnext101_32x8d | torchvision | 79.063 | 94.538 |
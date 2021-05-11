
# Benchmark - MobileNet

## Convert

The torchvision pretraining model can be transformed into zcls by script

```
$ cd /path/to/ZCls
$ python tools/converters/torchvision_mobilenet_to_zcls_mobilenet.py
```

## Benchmark

|     arch     |  framework  |  top1  |  top5  |
|:------------:|:-----------:|:------:|:------:|
| mobilenet_v2 |     zcls    | 71.429 | 90.151 |
| mobilenet_v2 | torchvision | 71.429 | 90.151 |
|  mnasnet0_5  |     zcls    | 66.961 | 86.946 |
|  mnasnet0_5  | torchvision | 66.961 | 86.946 |
|  mnasnet1_0  |     zcls    | 73.249 | 91.235 |
|  mnasnet1_0  | torchvision | 73.249 | 91.235 |
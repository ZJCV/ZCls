
# Benchmark - ShuffleNet

## Convert

The torchvision pretraining model can be transformed into zcls by script

```
$ cd /path/to/ZCls
$ python tools/converters/torchvision_shufflenet_to_zcls_shufflenet.py
```

## Benchmark

|        arch        |  framework  |  top1  |  top5  |
|:------------------:|:-----------:|:------:|:------:|
| shufflenet_v2_x0_5 |     zcls    | 59.735 | 81.222 |
| shufflenet_v2_x0_5 | torchvision | 59.735 | 81.226 |
| shufflenet_v2_x1_0 |     zcls    | 68.944 | 88.214 |
| shufflenet_v2_x1_0 | torchvision | 68.942 | 88.214 |
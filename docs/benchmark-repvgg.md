
# Benchmark - RepVGG

## Convert

The RepVGG pretraining model can be transformed into zcls by script

```
$ cd /path/to/ZCls
$ python tools/converters/official_repvgg_to_zcls_repvgg.py
```

## Benchmark

|        arch       | framework |  top1  |  top5  | input_size |  dataset |
|:-----------------:|:---------:|:------:|:------:|:----------:|:--------:|
|  repvgg_a0_train  |    zcls   | 72.009 | 90.389 |   224x224  | imagenet |
|  repvgg_a0_infer  |    zcls   | 72.007 | 90.389 |   224x224  | imagenet |
|  repvgg_a1_train  |    zcls   | 74.092 | 91.695 |   224x224  | imagenet |
|  repvgg_a1_infer  |    zcls   | 74.092 | 91.699 |   224x224  | imagenet |
|  repvgg_a2_train  |    zcls   | 76.286 | 93.034 |   224x224  | imagenet |
|  repvgg_a2_infer  |    zcls   | 76.284 | 93.036 |   224x224  | imagenet |
|  repvgg_b0_train  |    zcls   | 74.934 | 92.312 |   224x224  | imagenet |
|  repvgg_b0_infer  |    zcls   | 74.944 | 92.312 |   224x224  | imagenet |
|  repvgg_b1_train  |    zcls   | 78.127 | 94.182 |   224x224  | imagenet |
|  repvgg_b1_infer  |    zcls   | 78.121 | 94.182 |   224x224  | imagenet |
| repvgg_b1g2_train |    zcls   | 77.803 | 93.866 |   224x224  | imagenet |
| repvgg_b1g2_infer |    zcls   | 77.805 | 93.864 |   224x224  | imagenet |
| repvgg_b1g4_train |    zcls   | 77.579 | 93.726 |   224x224  | imagenet |
| repvgg_b1g4_infer |    zcls   | 77.579 | 93.726 |   224x224  | imagenet |
|  repvgg_b2_train  |    zcls   | 78.789 | 94.326 |   224x224  | imagenet |
|  repvgg_b2_infer  |    zcls   | 78.787 | 94.324 |   224x224  | imagenet |
| repvgg_b2g4_train |    zcls   | 79.491 | 94.768 |   224x224  | imagenet |
| repvgg_b2g4_infer |    zcls   | 79.495 | 94.768 |   224x224  | imagenet |
|  repvgg_b3_train  |    zcls   | 80.436 | 95.278 |   224x224  | imagenet |
|  repvgg_b3_infer  |    zcls   | 80.442 | 95.280 |   224x224  | imagenet |
| repvgg_b3g4_train |    zcls   | 80.070 | 95.154 |   224x224  | imagenet |
| repvgg_b3g4_infer |    zcls   | 80.072 | 95.156 |   224x224  | imagenet |
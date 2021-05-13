
# Pretrained Model

There are three scenarios using the pretraining model:

1. Using zcls pretraining model for reasoning
2. Using zcls pre training model for training
3. In the process of training, the program is interrupted and the training is resumed

## For reasoning

Configure the pretraining model path in the configuration file like this:

```
MODEL:
  ...
  ...
  RECOGNIZER:
    PRELOADED: "/path/to/pretrained"
```

## For training

Configure the pretraining model path in the configuration file like this:

```
MODEL:
  ...
  ...
  RECOGNIZER:
    PRETRAINED: "/path/to/pretrained"
```

## For resumed

Add the following fields `--resume` to the command line parameters：

```
$ CUDA_VISIBLE_DEVICES=3 python tools/test.py -cfg=configs/benchmarks/resnet/r18_zcls_imagenet_224.yaml --resume
```
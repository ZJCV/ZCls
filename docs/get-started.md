
# Get Started with ZCls

## How to train

1. Add dataset path to config_file, like CIFAR100

```
  NAME: 'CIFAR100'
  TRAIN_ROOT: './data/cifar'
  TEST_ROOT: './data/cifar'
```

*Note: current support `CIFAR10/CIFAR100/FashionMNIST/ImageNet`*

2. Add environment variable

```
$ export PYTHONPATH=/path/to/ZCls
```

3. Train

```
$ CUDA_VISIBLE_DEVICES=0 python tool/train.py -cfg=configs/benchmarks/r50_cifar100_224_e100_rmsprop.yaml
```

After training, the corresponding model can be found in `outputs/`, add model path to xxx.yaml

```
    PRELOADED: ""
```

4. Test

```
$ CUDA_VISIBLE_DEVICES=0 python tool/test.py -cfg=configs/benchmarks/r50_cifar100_224_e100_rmsprop.yaml
```

5. If finished the training halfway, resume it like this

```
$ CUDA_VISIBLE_DEVICES=0 python tool/train.py -cfg=configs/benchmarks/r50_cifar100_224_e100_rmsprop.yaml --resume
```

6. Use multiple GPU to train

```
$ CUDA_VISIBLE_DEVICES=0<,1,2,3> python tool/train.py -cfg=configs/benchmarks/r50_cifar100_224_e100_rmsprop.yaml -g=<N>
```

## How to add Dataset

Suppose your dataset is in the following format

```
root/dog/xxx.png
root/dog/xxy.png
root/dog/xxz.png

root/cat/123.png
root/cat/nsdf3.png
root/cat/asd932_.png
```

modify config_file like this

```
DATASET:
  NAME: 'GeneralDataset'
  TRAIN_ROOT: /path/to/train_root
  TEST_ROOT: /path/to/test_root
  TOP_K: (1, 5)
```

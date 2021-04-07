
# Use Builtin Datasets

zcls implements the following datasets:

1. [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)
2. [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)
3. [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist)
4. [ImageNet](http://www.image-net.org/)

## CIFAR10

Modify the configuration file as follows:

```
DATASET:
  NAME: 'CIFAR10'
  TRAIN_ROOT: /path/to/train_root
  TEST_ROOT: /path/to/test_root
  TOP_K: (1, 5)
```

*Note: If the path is not set correctly or the file does not exist, zcls will automatically download the dataset*

## CIFAR100

Modify the configuration file as follows:

```
DATASET:
  NAME: 'CIFAR100'
  TRAIN_ROOT: /path/to/train_root
  TEST_ROOT: /path/to/test_root
  TOP_K: (1, 5)
```

*Note: If the path is not set correctly or the file does not exist, zcls will automatically download the dataset*

## FashionMNIST

Modify the configuration file as follows:

```
DATASET:
  NAME: 'FashionMNIST'
  TRAIN_ROOT: /path/to/train_root
  TEST_ROOT: /path/to/test_root
  TOP_K: (1, 5)
```

*Note: If the path is not set correctly or the file does not exist, zcls will automatically download the dataset*

## ImageNet

Modify the configuration file as follows:

```
DATASET:
  NAME: 'ImageNet'
  TRAIN_ROOT: /path/to/train_root
  TEST_ROOT: /path/to/test_root
  TOP_K: (1, 5)
```

Download the following datasets to the data directory

```
.
├── ILSVRC2012_devkit_t12.tar.gz
├── ILSVRC2012_devkit_t3.tar.gz
├── ILSVRC2012_img_test.tar
├── ILSVRC2012_img_train_t3.tar
├── ILSVRC2012_img_train.tar
├── ILSVRC2012_img_val.tar
```

## Related references

* [CIFAR](https://pytorch.org/vision/stable/datasets.html#cifar)
* [Fashion-MNIST](https://pytorch.org/vision/stable/datasets.html#fashion-mnist)
* [ImageNet](https://pytorch.org/vision/stable/datasets.html#imagenet)
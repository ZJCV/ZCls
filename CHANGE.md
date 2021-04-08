# CHANGE

## v0.4.1

* New features
    1. Add mkdocs project
    2. Update README
    3. add tool/zoom.py
    4. add Grayscale transform
* Bug fixes
* Breaking changes.

## v0.4.0

* New features
    1. add GeneralDataset class
    2. add LMDBDataset class
    3. add LMDBImageNet class
    4. add README usage
* Bug fixes
    1. when use prefetcher in inference, release it after one epoch
    2. split train/test data path in config_file
* Breaking changes.

## v0.3.7

* New features
* Bug fixes
    1. update python requires in requirements.txt and setup.py
* Breaking changes.

## v0.3.6

* New features
    1. Add LMDB ImageNet
* Bug fixes
    1. one epoch end, use release() to del prefetcher
* Breaking changes.

## v0.3.5

* New features
* Bug fixes
    1. create prefetcher in every epoch
* Breaking changes.

## v0.3.4

* New features
    1. Add PREFETCHER
* Bug fixes
    1. Distinguish local or remote pretrained links
* Breaking changes.

## v0.3.3

* New features
    1. Add CIFAR10 dataset
* Bug fixes
    1. Fix label_smoothing_loss usage
* Breaking changes.

## v0.3.2

* New features
* Bug fixes
    1. When do_evaluation, need add test_data_loader
* Breaking changes.

## v0.3.1

* New features
    1. Change the loading method of test data set
* Bug fixes
* Breaking changes.

## v0.3.0

* New features
    1. Extract evaluator implementation, abstract out the general evaluator
    2. Command line parameter parse of merge training phase / test phase
    3. Refactoring the implementation of transforms module
* Bug fixes
    1. The Imagenet category entry is a tuple, not a string
* Breaking changes.

## v0.2.1

* New features
    1. Add FashionMNIST/ImageNet dataset
* Bug fixes
* Breaking changes.

## v0.2.0

* New features
    1. Add python package
* Bug fixes
* Breaking changes.

## v0.1.0

* New features
    1. Add CIFAR100 dataset
    2. Add Torchvision Transforms and AutoAugment
    3. Add batch dataloader
    4. Add multi-gpu training/testing
    5. Add hybrid precision training
    6. Add gradient accumulate training
    7. Add ResNet/ResNeXt/SKNet/ResNeSt/SENet/GCNet/Non-local/MobileNetV1-V2-V3/ShuffleNetV1-V2/MNASNet
    8. Add CrossEntropyLoss/LabelSmoothingLoss
    9. Add Adam/SGD/RMSProp
    10. Add GradualWarmup/CosineAnnealingLR/MultiStepLR
* Bug fixes
* Breaking changes.

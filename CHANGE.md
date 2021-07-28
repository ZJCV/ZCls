# CHANGE

## v0.10.1

* New features
    1. add new module: DiverseBranchBlock
    2. add tool: fuse block for ACBlock/RepVGGBLock/DBBlock
* Bug fixes
* Breaking changes.

## v0.10.0

* New features
    1. add RandomRotation/RandomErasing support
    2. add mixup/cutmix support
    3. add ghostnet attention module' sigmoid type setting
* Bug fixes
* Breaking changes.

## v0.9.3

* New features
* Bug fixes
    1. moduleNotFoundError: No module named 'resnest.torch.resnest'
* Breaking changes.

## v0.9.2

* New features
    1. add custom transform: SquarePad
    2. use torchvision.autoaugment replace ztransforms
* Bug fixes
    1. When some category data of the test set is empty, RuntimeError: stack expects a non-empty TensorList
* Breaking changes.

## v0.9.1

* New features
    1. update the accuracy calculation and calculate the TOPK accuracy of each category
    2. when compute dataset's accuracy, make assert for security
* Bug fixes
* Breaking changes.

## v0.9.0

* New features
* Bug fixes
    1. fix Dropout inplace operation make gradient computation failed
* Breaking changes.

## v0.8.0

* New features
    1. update hybrid_precision/distributed_data_parallel/gradient_acculmulate usage
    2. remove cfg.DATALOADER.SHUFFLE; use cfg.DATALOADER.RANDOM_SAMPLE
* Bug fixes
    1. add torch.cuda.empty_cache() to fix momory leak
    2. use cfg.MODEL.RECOGNIZER.PRETRAINED_NUM_CLASSES in head definition
* Breaking changes.

## v0.7.1

* New features
    1. add torchvision mobilenet_v3
    2. update parser.py usage
    3. update label_smoothing_loss usage
* Bug fixes
* Breaking changes.

## v0.7.0

* New features
    1. create EmptyLogger, used in subprocess
    2. refer torchvision to realize shufflenet_v2
    3. use nn.Hardswish replace custom HardswishWrapper
    4. add dropout config in mobilenet_v3 head
    5. add bias config in general_head_2d
    6. realize ghostnet
    7. upgrade development environment from torch 1.7.1 to 1.8.1
* Bug fixes
    1. fix the install requires bug
    2. fix resnet_d_backbone's fast_avg usage
* Breaking changes.

## v0.6.0

* New features
    1. transform repvgg/sknet pretrained model to zcls format
    2. update repvgg backbone and add attention module
    3. add bias config item for se-block
    4. open nn.Linear bias config for sk-block
    5. cancel bn2 for sknet_block
* Bug fixes
* Breaking changes.

## v0.5.2

* New features
    1. transform torchvision mobilenet/shufflenet pretrained model to zcls format
* Bug fixes
    1. use multiple mnanet/shufflenet model, deepcopy every stage_setting
* Breaking changes.

## v0.5.1

* New features
    1. transform torchvision resnet pretrained model to zcls format
    2. update pretrained model usage
* Bug fixes
* Breaking changes.

## v0.5.0

* New features
    1. Add docs for trick-data/trick-train
    2. Add public function get_classes for dataset
* Bug fixes
* Breaking changes.

## v0.4.2

* New features
    1. add DATALOADER SHUFFLE/RANDOM_SAMPLE config
    2. update lmdbdataset get image way
    3. update tools/zoom.py process way
* Bug fixes
    1. LMDBDataset: valueError: Decompressed Data Too Large
* Breaking changes.

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

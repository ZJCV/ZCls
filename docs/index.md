# ZCls(v0.13.3)

Welcome to zcls, a classification model training/inferring framework.

## Chapter

* [Installation](./install.md)
* [Roadmap](./roadmap.md)
* [Get Started with ZCls](./get-started.md)
* [Using Pretrained Model](./pretrained-model.md)
* [Model Zoo](./model-zoo.md)
* Dataset
    * [Use Builtin Datasets](./builtin-datasets.md)
    * Use Custom Datasets
        * [GeneralDataset](./general_dataset.md)
        * [LMDBDataset](./lmdb_dataset.md)
        * [MPDataset](./mp_dataset.md)
* Trick
    * [Data](./trick-data.md)
    * [Train](./trick-train.md)
* Benchmark
    * [ResNet/ResNeXt](./benchmark-resnet.md)
    * [SENet/SKNet/ResNeSt](./benchmark-resnest.md)
    * [MobileNet](./benchmark-mobilenet.md)
    * [ShuffleNet](./benchmark-shufflenet.md)
    * [RepVGG](./benchmark-repvgg.md)
    * [GhostNet](./benchmark-ghostnet.md)

## Breaking Changes

1. From `v0.13.0`, add [albumentation](https://github.com/albumentations-team/albumentations) to replace [torchvision](https://github.com/pytorch/vision) as the main transform backbend. Now the image format is `BGR` not `RGB`
# CHANGE

## v0.15.2

* New features
* Bug fixes
  * perf(torchvision): update load_state_dict_from_url usage
* Breaking changes.

## v0.15.1

* New features
* Bug fixes
  * fix(optim): filter all layers which require_grad=False
  * fix(dataloader): when using cpu, set pin_memory=False. Default: True
* Breaking changes.

## v0.15.0

* New features
  * perf(model): add load_pretrained_weights func
  * perf(base_recognizer.py): update _init_weights
  * feat(tools): add_md5_for_pths
  * perf(config): add cfg.DATASET.KEEP_RGB to keep RGB data format or not… 
    * Default is False
* Bug fixes
  * perf(checkpoint): adapt to the latest torchvision version 
    * prevent error: "from torch.utils.model_zoo import load_url as load_state_dict_from_url"
  * fix(heads): use cfg.MODEL.HEAD.NUM_CLASSES in head definition
  * fix(trainer.py): fix max_iter calculate
* Breaking changes.

## v0.14.0

* New features
  * perf(transforms): update Resize/AutoAugment/SquarePad
    1. add largest edge mode for Resize;
    2. optimize realization for Resize/AutoAugment/SquarePad.
* Bug fixes
  * fix(transforms): support Resize and Resize2 together
* Breaking changes.

## v0.13.6

* New features
  * perf(color_jitter.py): add hue use
  * perf(resize.py): update Resize realization and usage
    1. Supports scaling based on a single dimension; 
    2. Added a new zoom option Resize2, for secondary scaling
* Bug fixes
* Breaking changes.

## v0.13.5

* New features
  * perf(base_evaluator.py): update result_str format
  * perf(color_jitter.py): add hue config
* Bug fixes
  * fix(mpdataset): fix data read index order
  * fix(configs): fix TRANSFORMS order: ('ToTensor', 'Normalize') to ('Normalize', 'ToTensor')
* Breaking changes.

## v0.13.4

* New features
  * perf(evaluator): update topk output format
* Bug fixes
  * fix(dataset): fix default_converter usage
* Breaking changes.

## v0.13.3

* New features
  * perf(mp_dataset.py): change file format and enhance the loading way
* Bug fixes
* Breaking changes.

## v0.13.2

* New features
* Bug fixes
  1. fix(cv2.cvtColor): Argument 'code' is required to be an integer 
  2. fix(datasets): when convert PIL.Image to np.ndarray, synchronization settings RGB2BGR
* Breaking changes.

## v0.13.1

* New features
* Bug fixes
  1. fix(datasets): use np.ndarray instead of PIL.Image to preprocess image
* Breaking changes.

## v0.13.0

* New features
    1. feat(transforms): use albumentation replace torchvision as backbend
* Bug fixes
* Breaking changes.
    1. use albumentation replace torchvision as backbend

## v0.12.0

* New features
    1. feat(dataset): add GeneralDatasetV2
    2. feat(dataset): add MPDataset for large-scale data loading
    3. perf(dataloader): in train phase, keep drop_last=True
    4. feat(sampler): custom DistributedSampler used for IterableDataset
    5. Update issue templates
* Bug fixes
    1. fix(inference.py): add KEY_OUTPUT import
* Breaking changes.

## v0.11.1

* New features
* Bug fixes
    1. fix(inference): only use output_dict[KEY_OUTPUT] when infering
    2. fix(transform): use cfg.TRANSFORM.KEEP_BITS instead of cfg.TRANSFORM.BITS
* Breaking changes.

## v0.11.0

* New features
    1. feat(transform): custom resize using opencv replacing pil
* Bug fixes
    1. fix(checkpoint): when resume, make cur_epoch + 1
* Breaking changes.

## v0.10.3

* New features
    1. new image processing strategy: RandomAutocontrast/RandomAdjustSharpness/RandomPosterize/ToPILImage
* Bug fixes
* Breaking changes.

## v0.10.2

* New features
    1. add gradient_clip feature
    2. add init weights for DDB
* Bug fixes
    1. input model.parameters() to clip_grad_norm_ rather than model
* Breaking changes.

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

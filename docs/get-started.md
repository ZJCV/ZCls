
# Get Started with ZCls

1. Add dataset path to config_file, like CIFAR100

        NAME: 'CIFAR100'
        TRAIN_ROOT: './data/cifar'
        TEST_ROOT: './data/cifar'

      *Note 1: current support `CIFAR10/CIFAR100/FashionMNIST/ImageNet`*

      *Note 2: use `BGR` image format*

2. Add environment variable

        $ export PYTHONPATH=/path/to/ZCls

3. Train

        $ CUDA_VISIBLE_DEVICES=0 python tool/train.py -cfg=configs/cifar/r50_cifar100_224_e100_rmsprop.yaml

      After training, the corresponding model can be found in `outputs/`

4. Using pretrained model, refer to [Pretrained Model](./pretrained-model.md)

5. If finished the training halfway, resume it like this

        $ CUDA_VISIBLE_DEVICES=0 python tool/train.py -cfg=configs/cifar/r50_cifar100_224_e100_rmsprop.yaml --resume

6. Use multiple GPU to train

        $ CUDA_VISIBLE_DEVICES=0<,1,2,3> python tool/train.py -cfg=configs/cifar/r50_cifar100_224_e100_rmsprop.yaml -g=<N>
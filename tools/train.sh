#!/bin/bash

export PYTHONPATH=.

#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port="16231" \
#        tools/train.py -a resnet18 --pretrained --b 256 --workers 4 --opt-level O1 \
#        /home/zj/data/cifar

#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port="16231" \
#        tools/train.py -a resnet18 --pretrained --b 256 --workers 4 --dataset mp --opt-level O1 \
#        /home/zj/data/cifar/zcls_cifar

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port="16231" \
        tools/train.py -a resnet18 --pretrained --b 256 --workers 4 --dataset general_v2 --opt-level O1 \
        /home/zj/data/cifar/zcls_cifar
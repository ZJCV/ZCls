# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午1:57
@file: build.py
@author: zj
@description: 
"""

from .resnet import get_resnet
from .ghostnet import get_ghostnet

from zcls.util import logging

logger = logging.get_logger(__name__)

__supported_model__ = [
    'resnet18',
    'ghostnet_130',
]


def build_model(args, memory_format):
    assert args.arch in __supported_model__, f"{args.arch} do not in model list"

    # create model
    if args.pretrained:
        logger.info("=> using pre-trained model '{}'".format(args.arch))
    else:
        logger.info("=> creating model '{}'".format(args.arch))

    if 'ghostnet' in args.arch:
        model = get_ghostnet(pretrained=args.pretrained, num_classes=args.num_classes, arch=args.arch)
    elif 'resnet' in args.arch:
        model = get_resnet(pretrained=args.pretrained, num_classes=args.num_classes, arch=args.arch)
    else:
        raise ValueError(f"{args.arch} does not support")

    if args.sync_bn:
        import apex
        logger.info("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    model = model.cuda().to(memory_format=memory_format)

    return model

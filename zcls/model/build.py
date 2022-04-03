# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午1:57
@file: build.py
@author: zj
@description: 
"""

import torchvision.models as models


def build_model(args, memory_format):
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if args.sync_bn:
        import apex
        print("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    model = model.cuda().to(memory_format=memory_format)

    return model

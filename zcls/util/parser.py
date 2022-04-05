# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午1:30
@file: parser.py
@author: zj
@description: 
"""

import os
import argparse

from ..model.model.build import __supported_model__
from ..data.dataset.build import __supported_dataset__


def parse():
    model_names = sorted(__supported_model__)
    dataset_names = sorted(__supported_dataset__)

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--dataset', '-d', metavar='DATASET', default='general',
                        choices=dataset_names,
                        help='dataset type: ' +
                             ' | '.join(dataset_names) +
                             ' (default: general)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet18)')
    parser.add_argument('-n', '--num-classes', default=1000, type=int, metavar='NUM-CLASSES',
                        help='number of model output (default: 1000)')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size per process (default: 256)')

    parser.add_argument('-l', '--loss', default="CrossEntropyLoss", type=str, metavar='LOSS',
                        help='Loss type (default: CrossEntropyLoss)')

    parser.add_argument('-optim', '--optimizer', default="sgd", type=str, metavar='OPTIM',
                        help='Optimizer type (default: sgd)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR',
                        help='Initial learning rate.  Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256.  A warmup schedule will also be applied over the first 5 epochs.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--warmup', default=True, type=bool, metavar='WARMUP',
                        help='Is warmup (default: True)')
    parser.add_argument('--warmup-epochs', default=5, type=int, metavar='WARMUP-EPOCHS',
                        help='Warmup epochs (default: 5)')
    parser.add_argument('--lr-scheduler', default='MultiStepLR', type=str, metavar='LR-SCHEDULER',
                        help='LR scheduler type (default: MultiStepLR)')

    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--output-dir', '-o', default='outputs', type=str,
                        metavar='OUTPUT_DIR', help='output path (default: outputs)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')

    parser.add_argument('--prof', default=-1, type=int,
                        help='Only run 10 iterations for profiling.')
    parser.add_argument('--deterministic', action='store_true')

    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', 0), type=int)
    parser.add_argument('--sync_bn', action='store_true',
                        help='enabling apex sync BN.')

    parser.add_argument('--opt-level', type=str)
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--loss-scale', type=str, default=None)
    parser.add_argument('--channels-last', type=bool, default=False)
    args = parser.parse_args()

    return args

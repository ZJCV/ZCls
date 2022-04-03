# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午1:40
@file: infer.py
@author: zj
@description: 
"""

import torch
import time

from ..util.meter import AverageMeter
from ..util.prefetcher import data_prefetcher
from ..util.metric import accuracy
from ..util.distributed import reduce_tensor
from ..util.misc import to_python_float

from zcls.util import logging

logger = logging.get_logger(__name__)


def validate(args, val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    prefetcher = data_prefetcher(val_loader)
    input, target = prefetcher.next()
    i = 0
    while input is not None:
        i += 1

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(args.world_size, loss.data)
            prec1 = reduce_tensor(args.world_size, prec1)
            prec5 = reduce_tensor(args.world_size, prec5)
        else:
            reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # TODO:  Change timings to mirror train().
        if args.local_rank == 0 and i % args.print_freq == 0:
            logger.info('Test: [{0}/{1}]\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Speed {2:.3f} ({3:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader),
                args.world_size * args.batch_size / batch_time.val,
                args.world_size * args.batch_size / batch_time.avg,
                batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

        input, target = prefetcher.next()

    logger.info(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
           .format(top1=top1, top5=top5))

    return top1.avg

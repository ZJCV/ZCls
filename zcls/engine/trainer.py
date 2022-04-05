# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午1:34
@file: trainer.py
@author: zj
@description: 
"""

import time
import torch

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

from ..util.meter import AverageMeter
from ..util.prefetcher import data_prefetcher
from ..util.metric import accuracy
from ..util.distributed import reduce_tensor
from ..util.misc import to_python_float
from ..optim.lr_scheduler.build import adjust_learning_rate

from zcls.util import logging

logger = logging.get_logger(__name__)


def train(args, train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    prefetcher = data_prefetcher(train_loader)
    input, target = prefetcher.next()
    i = 0
    while input is not None:
        i += 1
        if args.prof >= 0 and i == args.prof:
            logger.info("Profiling begun at iteration {}".format(i))
            torch.cuda.cudart().cudaProfilerStart()

        if args.prof >= 0: torch.cuda.nvtx.range_push("Body of iteration {}".format(i))

        adjust_learning_rate(args, optimizer, epoch, i, len(train_loader))

        # compute output
        if args.prof >= 0: torch.cuda.nvtx.range_push("forward")
        output = model(input)
        if args.prof >= 0: torch.cuda.nvtx.range_pop()
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()

        if args.prof >= 0: torch.cuda.nvtx.range_push("backward")
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        # for param in model.parameters():
        #     print(param.data.double().sum().item(), param.grad.data.double().sum().item())

        if args.prof >= 0: torch.cuda.nvtx.range_push("optimizer.step()")
        optimizer.step()
        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        if i % args.print_freq == 0:
            # Every print_freq iterations, check the loss, accuracy, and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.

            # Measure accuracy
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            # Average loss and accuracy across processes for logging
            if args.distributed:
                reduced_loss = reduce_tensor(args.world_size, loss.data)
                prec1 = reduce_tensor(args.world_size, prec1)
                prec5 = reduce_tensor(args.world_size, prec5)
            else:
                reduced_loss = loss.data

            # to_python_float incurs a host<->device sync
            losses.update(to_python_float(reduced_loss), input.size(0))
            top1.update(to_python_float(prec1), input.size(0))
            top5.update(to_python_float(prec5), input.size(0))

            torch.cuda.synchronize()
            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()

            if args.local_rank == 0:
                logger.info('Epoch: [{0}][{1}/{2}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Speed {3:.3f} ({4:.3f})\t'
                            'Lr {lr:.4f}\t'
                            'Loss {loss.val:.10f} ({loss.avg:.4f})\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(train_loader),
                    args.world_size * args.batch_size / batch_time.val,
                    args.world_size * args.batch_size / batch_time.avg,
                    batch_time=batch_time,
                    lr=optimizer.param_groups[0]['lr'],
                    loss=losses, top1=top1, top5=top5))
        if args.prof >= 0: torch.cuda.nvtx.range_push("prefetcher.next()")
        input, target = prefetcher.next()
        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        # Pop range "Body of iteration {}".format(i)
        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        if args.prof >= 0 and i == args.prof + 10:
            logger.info("Profiling ended at iteration {}".format(i))
            torch.cuda.cudart().cudaProfilerStop()
            quit()

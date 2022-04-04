import os

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from zcls.data.build import build_data
from zcls.optim.optimizer.build import build_optimizer
from zcls.model.criterion.build import build_criterion
from zcls.model.model.build import build_model
from zcls.engine.trainer import train
from zcls.engine.infer import validate
from zcls.util.parser import parse
from zcls.util.checkpoint import save_checkpoint
from zcls.data.dataset.mp_dataset import MPDataset

from zcls.util import logging

logger = logging.get_logger(__name__)

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


def main():
    global best_prec1, args

    args = parse()

    cudnn.benchmark = True
    best_prec1 = 0
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.local_rank)
        torch.set_printoptions(precision=10)

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    args.output_dir = 'outputs'
    if args.local_rank == 0 and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logging.setup_logging(local_rank=args.local_rank, output_dir=args.output_dir)
    logger.info(args)

    logger.info("opt_level = {}".format(args.opt_level))
    logger.info("keep_batchnorm_fp32 = {}".format(args.keep_batchnorm_fp32, type(args.keep_batchnorm_fp32)))
    logger.info("loss_scale = {}".format(args.loss_scale, type(args.loss_scale)))

    logger.info("CUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))

    if args.channels_last:
        memory_format = torch.channels_last
    else:
        memory_format = torch.contiguous_format

    model = build_model(args, memory_format)

    # Scale learning rate based on global batch size
    args.lr = args.lr * float(args.batch_size * args.world_size) / 256.
    optimizer = build_optimizer(args, model)

    # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.
    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                      loss_scale=args.loss_scale
                                      )

    # For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
    # This must be done AFTER the call to amp.initialize.  If model = DDP(model) is called
    # before model, ... = amp.initialize(model, ...), the call to amp.initialize may alter
    # the types of model's parameters in a way that disrupts or destroys DDP's allreduce hooks.
    if args.distributed:
        # By default, apex.parallel.DistributedDataParallel overlaps communication with
        # computation in the backward pass.
        # model = DDP(model)
        # delay_allreduce delays all communication to the end of the backward pass.
        model = DDP(model, delay_allreduce=True)

    # define loss function (criterion) and optimizer
    criterion = build_criterion(args).cuda()

    args.epoch = 0
    # Optionally resume from a checkpoint
    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume):
                logger.info("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(args.gpu))
                args.start_epoch = checkpoint['epoch']
                global best_prec1
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                logger.info("=> loaded checkpoint '{}' (epoch {})"
                            .format(args.resume, checkpoint['epoch']))
                args.epoch = checkpoint['epoch']
            else:
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

        resume()

    # # Data loading code
    train_sampler, train_loader, val_loader = build_data(args, memory_format)

    if args.evaluate:
        validate(args, val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if isinstance(train_loader.dataset, MPDataset):
            train_loader.dataset.set_epoch(epoch)
        elif args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(args, train_loader, model, criterion, optimizer, epoch)
        torch.cuda.empty_cache()

        # evaluate on validation set
        prec1 = validate(args, val_loader, model, criterion)
        torch.cuda.empty_cache()

        # remember best prec@1 and save checkpoint
        if args.local_rank == 0:
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best, output_dir=args.output_dir, filename=f'checkpoint_{epoch}.pth.tar')


if __name__ == '__main__':
    main()

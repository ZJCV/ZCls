# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午8:00
@file: trainer.py
@author: zj
@description: 
"""

import os
import datetime
import time
import torch
from torch.nn.parallel import DistributedDataParallel

from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

from zcls.config.key_word import KEY_LOSS
from zcls.util.metric_logger import MetricLogger, update_stats, log_iter_stats, log_epoch_stats
from zcls.util.precise_bn import calculate_and_update_precise_bn
from zcls.util.distributed import is_master_proc, synchronize
from zcls.util import logging
from zcls.util.prefetcher import Prefetcher
from zcls.engine.inference import do_evaluation
from zcls.data.build import shuffle_dataset

logger = logging.get_logger(__name__)


def do_train(cfg, arguments,
             train_data_loader, test_data_loader,
             model, criterion, optimizer, lr_scheduler,
             check_pointer, device):
    meters = MetricLogger()
    evaluator = train_data_loader.dataset.evaluator
    summary_writer = None
    use_tensorboard = cfg.TRAIN.USE_TENSORBOARD
    if is_master_proc() and use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        summary_writer = SummaryWriter(log_dir=os.path.join(cfg.OUTPUT_DIR, 'tf_logs'))

    log_step = cfg.TRAIN.LOG_STEP
    save_epoch = cfg.TRAIN.SAVE_EPOCH
    eval_epoch = cfg.TRAIN.EVAL_EPOCH
    max_epoch = cfg.TRAIN.MAX_EPOCH
    gradient_accumulate_step = cfg.TRAIN.GRADIENT_ACCUMULATE_STEP

    start_epoch = arguments['cur_epoch']
    epoch_iters = len(train_data_loader)
    max_iter = (max_epoch - start_epoch) * epoch_iters
    current_iterations = 0

    # Creates a GradScaler once at the beginning of training.
    scalar = GradScaler()
    model.train()
    optimizer.zero_grad()

    synchronize()
    logger.info("Start training ...")
    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch))
    start_training_time = time.time()
    end = time.time()
    for cur_epoch in range(start_epoch, max_epoch + 1):
        shuffle_dataset(train_data_loader, cur_epoch, is_shuffle=cfg.DATALOADER.RANDOM_SAMPLE)
        data_loader = Prefetcher(train_data_loader, device) if cfg.DATALOADER.PREFETCHER else train_data_loader
        for iteration, (images, targets) in enumerate(data_loader):
            if not cfg.DATALOADER.PREFETCHER:
                images = images.to(device=device, non_blocking=True)
                targets = targets.to(device=device, non_blocking=True)

            if cfg.TRAIN.HYBRID_PRECISION:
                # Runs the forward pass with autocasting.
                with autocast():
                    output_dict = model(images)
                    loss_dict = criterion(output_dict, targets)
                    loss = loss_dict[KEY_LOSS] / gradient_accumulate_step
                if isinstance(model, DistributedDataParallel):
                    # multi-gpu distributed training
                    with model.no_sync():
                        scalar.scale(loss).backward()
                else:
                    scalar.scale(loss).backward()
            else:
                output_dict = model(images)
                loss_dict = criterion(output_dict, targets)
                loss = loss_dict[KEY_LOSS] / gradient_accumulate_step

                if isinstance(model, DistributedDataParallel):
                    # multi-gpu distributed training
                    with model.no_sync():
                        loss.backward()
                else:
                    loss.backward()

            current_iterations += 1
            if current_iterations % gradient_accumulate_step == 0:
                current_iterations = 0
                if cfg.TRAIN.HYBRID_PRECISION:
                    scalar.step(optimizer)
                    scalar.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            acc_list = evaluator.evaluate_train(output_dict, targets)
            update_stats(cfg.NUM_GPUS, meters, loss_dict, acc_list)

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time)
            if (iteration + 1) % log_step == 0:
                logger.info(log_iter_stats(
                    iteration, epoch_iters, cur_epoch, max_epoch, optimizer.param_groups[0]['lr'], meters))
            if is_master_proc() and summary_writer:
                global_step = (cur_epoch - 1) * epoch_iters + (iteration + 1)
                for name, meter in meters.meters.items():
                    summary_writer.add_scalar('{}/avg'.format(name), float(meter.avg),
                                              global_step=global_step)
                    summary_writer.add_scalar('{}/global_avg'.format(name), meter.global_avg,
                                              global_step=global_step)
                summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)

        if cfg.DATALOADER.PREFETCHER:
            data_loader.release()
            del data_loader
        torch.cuda.empty_cache()
        logger.info(log_epoch_stats(epoch_iters, cur_epoch, max_epoch, optimizer.param_groups[0]['lr'], meters))
        arguments["cur_epoch"] = cur_epoch
        lr_scheduler.step()
        if is_master_proc() and save_epoch > 0 and cur_epoch % save_epoch == 0 and cur_epoch != max_epoch:
            check_pointer.save("model_{:04d}".format(cur_epoch), **arguments)
        if eval_epoch > 0 and cur_epoch % eval_epoch == 0 and cur_epoch != max_epoch:
            if cfg.MODEL.NORM.PRECISE_BN:
                calculate_and_update_precise_bn(
                    train_data_loader,
                    model,
                    min(cfg.MODEL.NORM.NUM_BATCHES_PRECISE, len(train_data_loader)),
                    cfg.NUM_GPUS > 0,
                )

            eval_results = do_evaluation(cfg, model, test_data_loader, device, cur_epoch=cur_epoch)
            model.train()
            if is_master_proc() and summary_writer:
                for key, value in eval_results.items():
                    summary_writer.add_scalar(f'eval/{key}', value, global_step=cur_epoch + 1)

    if eval_epoch > 0:
        logger.info('Start final evaluating...')
        torch.cuda.empty_cache()  # speed up evaluating after training finished
        eval_results = do_evaluation(cfg, model, test_data_loader, device)

        if is_master_proc() and summary_writer:
            for key, value in eval_results.items():
                summary_writer.add_scalar(f'eval/{key}', value, global_step=arguments["cur_epoch"])
            summary_writer.close()
    if is_master_proc():
        check_pointer.save("model_final", **arguments)
    # compute training time
    total_training_time = int(time.time() - start_training_time)
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / max_iter))
    return model

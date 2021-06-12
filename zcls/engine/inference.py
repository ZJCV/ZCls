# -*- coding: utf-8 -*-

"""
@date: 2020/8/23 上午9:51
@file: inference.py
@author: zj
@description: 
"""

import os
import datetime
import torch
from tqdm import tqdm

import zcls.util.logging as logging
from zcls.util.prefetcher import Prefetcher
from zcls.util.distributed import all_gather, is_master_proc

logger = logging.get_logger(__name__)


@torch.no_grad()
def compute_on_dataset(images, targets, model, num_gpus, evaluator):
    output_dict = model(images)
    # Gather all the predictions across all the devices to perform ensemble.
    if num_gpus > 1:
        keys = list()
        values = list()
        for key in sorted(output_dict):
            keys.append(key)
            values.append(output_dict[key])
        values = all_gather(values)
        output_dict = {k: v for k, v in zip(keys, values)}
        targets = all_gather([targets])[0]

    evaluator.evaluate_test(output_dict, targets)


@torch.no_grad()
def inference(cfg, model, test_data_loader, device, **kwargs):
    cur_epoch = kwargs.get('cur_epoch', None)
    dataset_name = cfg.DATASET.NAME
    num_gpus = cfg.NUM_GPUS

    dataset = test_data_loader.dataset
    evaluator = test_data_loader.dataset.evaluator
    evaluator.clean()

    logger.info("Evaluating {} dataset({} video clips):".format(dataset_name, len(dataset)))

    data_loader = Prefetcher(test_data_loader, device) if cfg.DATALOADER.PREFETCHER else test_data_loader
    if is_master_proc():
        for images, targets in tqdm(data_loader):
            if not cfg.DATALOADER.PREFETCHER:
                images = images.to(device=device, non_blocking=True)
                targets = targets.to(device=device, non_blocking=True)
            compute_on_dataset(images, targets, model, num_gpus, evaluator)
    else:
        for images, targets in data_loader:
            if not cfg.DATALOADER.PREFETCHER:
                images = images.to(device=device, non_blocking=True)
                targets = targets.to(device=device, non_blocking=True)
            compute_on_dataset(images, targets, model, num_gpus, evaluator)

    if cfg.DATALOADER.PREFETCHER:
        data_loader.release()
    result_str, acc_dict = evaluator.get()
    logger.info(result_str)

    if is_master_proc():
        output_dir = cfg.OUTPUT_DIR
        result_path = os.path.join(output_dir,
                                   'result_{}.txt'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))) \
            if cur_epoch is None else os.path.join(output_dir, 'result_{:04d}.txt'.format(cur_epoch))

        with open(result_path, "w") as f:
            f.write(result_str)

    return acc_dict


@torch.no_grad()
def do_evaluation(cfg, model, test_data_loader, device, **kwargs):
    model.eval()

    eval_results = inference(cfg, model, test_data_loader, device, **kwargs)
    torch.cuda.empty_cache()
    return eval_results

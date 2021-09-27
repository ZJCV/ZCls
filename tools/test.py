# -*- coding: utf-8 -*-

"""
@date: 2020/9/18 上午9:15
@file: tests.py
@author: zj
@description: 
"""

import os
import numpy as np
import torch

from zcls.config import cfg
from zcls.data.build import build_data
from zcls.model.recognizers.build import build_recognizer
from zcls.engine.inference import do_evaluation
from zcls.util.collect_env import collect_env_info
from zcls.util import logging
from zcls.util.distributed import get_device, get_local_rank
from zcls.util.parser import parse_args, load_config
from zcls.util.misc import launch_job
from zcls.util.distributed import synchronize, init_distributed_training

logger = logging.get_logger(__name__)


def test(cfg):
    # Set up environment.
    init_distributed_training(cfg)
    local_rank_id = get_local_rank()

    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED + 10 * local_rank_id)
    torch.manual_seed(cfg.RNG_SEED + 10 * local_rank_id)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    logging.setup_logging(cfg.OUTPUT_DIR)

    device = get_device(local_rank=local_rank_id)
    model = build_recognizer(cfg, device=device)

    test_data_loader = build_data(cfg, is_train=False, rank_id=local_rank_id)

    synchronize()
    do_evaluation(cfg, model, test_data_loader, device)


def main():
    args = parse_args()
    load_config(args, cfg)

    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    logging.setup_logging(cfg.OUTPUT_DIR)
    logger.info(args)

    logger.info("Environment info:\n" + collect_env_info())
    logger.info("Loaded configuration file {}".format(args.config_file))
    if args.config_file:
        with open(args.config_file, "r") as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    launch_job(cfg=cfg, init_method=cfg.INIT_METHOD, func=test)


if __name__ == '__main__':
    main()

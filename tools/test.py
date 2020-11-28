# -*- coding: utf-8 -*-

"""
@date: 2020/9/18 上午9:15
@file: tests.py
@author: zj
@description: 
"""

import numpy as np
import torch

from zcls.model.recognizers.build import build_recognizer
from zcls.engine.inference import do_evaluation
from zcls.util.collect_env import collect_env_info
from zcls.util import logging
from zcls.util.distributed import get_device, get_local_rank
from zcls.util.parser import parse_test_args, load_test_config
from zcls.util.misc import launch_job
from zcls.util.distributed import synchronize, init_distributed_training

logger = logging.get_logger(__name__)


def test(cfg):
    # Set up environment.
    init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    logging.setup_logging(cfg.OUTPUT_DIR)

    device = get_device(local_rank=get_local_rank())
    model = build_recognizer(cfg, device=device)

    synchronize()
    do_evaluation(cfg, model, device)


def main():
    args = parse_test_args()
    cfg = load_test_config(args)

    logging.setup_logging(cfg.OUTPUT_DIR)
    logger.info(args)

    logger.info("Environment info:\n" + collect_env_info())
    logger.info("Loaded configuration file {}".format(args.config_file))
    if args.config_file:
        with open(args.config_file, "r") as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    launch_job(cfg=cfg, init_method=args.init_method, func=test)


if __name__ == '__main__':
    main()

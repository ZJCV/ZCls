#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Logging."""

import builtins
import logging
import os
import sys

import pycls.util.distributed as du


def _suppress_print():
    """
    Suppresses printing from the current process.
    """

    def print_pass(*objects, sep=" ", end="\n", file=sys.stdout, flush=False):
        pass

    builtins.print = print_pass


def setup_logging(name, output_dir=None):
    """
    Sets up the logging for multiple processes. Only enable the logging for the
    master process, and suppress logging for the non-master processes.
    """
    if not du.is_master_proc(du.get_world_size()):
        # Suppress logging for non-master processes.
        _suppress_print()
        logger = NllLogger(f'{name}.{du.get_rank()}')
        return logger

    logger = logging.getLogger(name)
    logging.root.handlers = []
    for handler in logger.handlers:
        logger.removeHandler(handler)

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    plain_formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(filename)s: %(lineno)3d: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(plain_formatter)
    logger.addHandler(ch)

    if output_dir:
        fh = logging.FileHandler(os.path.join(output_dir, 'log.txt'))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    return logger


def get_logger(name):
    """
    Retrieve the logger with the specified name or, if name is None, return a
    logger which is the root logger of the hierarchy.
    Args:
        name (string): name of the logger.
    """
    return logging.getLogger(name)


class NllLogger:

    def __init__(self, name: str) -> None:
        print('nll', name)
        self.logger = logging.getLogger(name)

    def info(self, msg, *args, **kwargs) -> None:
        print(msg)
        pass

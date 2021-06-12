# -*- coding: utf-8 -*-

"""
@date: 2020/10/4 下午3:09
@file: parser.py
@author: zj
@description: 
"""

import argparse
from yacs.config import CfgNode as CN


def parse_args():
    parser = argparse.ArgumentParser(description='ZCls Training/Test With PyTorch')
    parser.add_argument('-cfg',
                        "--config_file",
                        type=str,
                        default="",
                        metavar="FILE",
                        help="path to config file")
    parser.add_argument('--pretrained',
                        type=str,
                        default="",
                        metavar='PRETRAINED_FILE',
                        help="path to pretrained model")
    parser.add_argument('-out',
                        '--output_dir',
                        type=str,
                        default="",
                        metavar="OUTPUT_DIR",
                        help="path to output")

    parser.add_argument('--log_step',
                        type=int,
                        default=-1,
                        help='Print logs every log_step (default: 10)')
    parser.add_argument('--save_step',
                        type=int,
                        default=-1,
                        help='Save checkpoint every save_step, disabled when save_step < 0 (default: 1000)')
    parser.add_argument('--eval_step',
                        type=int,
                        default=-1,
                        help='Evaluate dataset every eval_step, disabled when eval_step < 0 (default: 1000)')

    parser.add_argument('--resume',
                        default=False,
                        action='store_true',
                        help='Resume training')
    parser.add_argument('--use_tensorboard',
                        default=True,
                        action='store_false')

    parser.add_argument('-g',
                        '--gpus',
                        type=int,
                        default=-1,
                        help='number of gpus per node (default: 1)')
    parser.add_argument('-n',
                        '--nodes',
                        type=int,
                        default=-1,
                        metavar='N',
                        help='number of machines (default: 1)')
    parser.add_argument('-nr',
                        '--nr',
                        type=int,
                        default=-1,
                        help='ranking within the nodes (default: 0)')
    parser.add_argument("--init_method",
                        help="Initialization method, includes TCP or shared file-system",
                        default="",
                        type=str)

    parser.add_argument("opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


def load_config(args, cfg):
    assert isinstance(args, argparse.Namespace)
    assert isinstance(cfg, CN)
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.pretrained:
        cfg.MODEL.RECOGNIZER.PRELOADED = args.pretrained
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.log_step != -1:
        cfg.TRAIN.LOG_STEP = args.log_step
    if args.save_step != -1:
        cfg.TRAIN.SAVE_STEP = args.save_step
    if args.eval_step != -1:
        cfg.TRAIN.EVAL_STEP = args.eval_step

    if args.resume:
        cfg.TRAIN.RESUME = True
    if not args.use_tensorboard:
        cfg.TRAIN.USE_TENSORBOARD = False

    if args.gpus != -1:
        cfg.NUM_GPUS = args.gpus
    if args.nodes != -1:
        cfg.NUM_NODES = args.nodes
    if args.nr != -1:
        cfg.RANK_ID = args.nr
    if args.init_method:
        cfg.INIT_METHOD = args.init_method

    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

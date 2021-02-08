# -*- coding: utf-8 -*-

"""
@date: 2020/10/4 下午3:09
@file: parser.py
@author: zj
@description: 
"""

import os
import argparse
from zcls.config import cfg


def parse_train_args():
    parser = argparse.ArgumentParser(description='PyCls Training With PyTorch')
    parser.add_argument('-cfg',
                        "--config_file",
                        type=str,
                        default="",
                        metavar="FILE",
                        help="path to config file")

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
                        default="tcp://localhost:39129",
                        type=str)

    parser.add_argument("opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


def parse_test_args():
    parser = argparse.ArgumentParser(description='PyCls Test With PyTorch')
    parser.add_argument("config_file",
                        type=str,
                        default="",
                        metavar="FILE",
                        help="path to config file")
    parser.add_argument('pretrained', default="", metavar='PRETRAINED_FILE',
                        help="path to pretrained model", type=str)
    parser.add_argument('--output', default="./outputs/tests", type=str)

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
                        default="tcp://localhost:39129",
                        type=str)

    args = parser.parse_args()
    return args


def load_train_config(args):
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
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
        # 在多gpu训练/测试中，同步增加学习率和批量大小
        cfg.OPTIMIZER.LR *= args.gpus
    if args.nodes != -1:
        cfg.NUM_NODES = args.nodes
    if args.nr != -1:
        cfg.RANK_ID = args.nr

    cfg.freeze()

    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    return cfg


def load_test_config(args):
    if not os.path.isfile(args.config_file) or not os.path.isfile(args.pretrained):
        raise ValueError('需要输入配置文件和预训练模型路径')

    cfg.merge_from_file(args.config_file)
    cfg.MODEL.RECOGNIZER.PRELOADED = args.pretrained
    cfg.OUTPUT_DIR = args.output

    if args.gpus != -1:
        cfg.NUM_GPUS = args.gpus
    if args.nodes != -1:
        cfg.NODES = args.nodes
    if args.nr != -1:
        cfg.RANK = args.nr
    cfg.freeze()

    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    return cfg

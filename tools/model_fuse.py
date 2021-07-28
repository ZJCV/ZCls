# -*- coding: utf-8 -*-

"""
@date: 2021/7/28 下午10:10
@file: model_fuse.py
@author: zj
@description: Fuse block for ACBlock/RepVGGBLock/DBBlock
"""

import argparse
import os.path

import torch

from zcls.model.conv_helper import fuse_acblock, fuse_repvgg_block, fuse_dbblock

from zcls.config import cfg
from zcls.model.recognizers.build import build_recognizer
from zcls.util.checkpoint import CheckPointer


def parse_args():
    parser = argparse.ArgumentParser(description='Fuse block for ACBlock/RepVGGBLock/DBBlock')
    parser.add_argument("config_file",
                        type=str,
                        default="",
                        metavar="CONFIG_FILE",
                        help="path to config file")
    parser.add_argument('output_dir',
                        type=str,
                        default="",
                        metavar="OUTPUT_DIR",
                        help="path to output")
    parser.add_argument('--verbose',
                        default=False,
                        action='store_true',
                        help="Print Model Info")

    args = parser.parse_args()
    return args


def fuse(cfg, model, verbose=False):
    block_name_tuple = cfg.MODEL.CONV.ADD_BLOCKS
    if block_name_tuple is not None:
        assert isinstance(block_name_tuple, tuple)
        # The insertion and fusion operations are in reverse order
        for add_block in list(reversed(block_name_tuple)):
            if verbose:
                print(f'fuse {add_block} ...')
            if add_block == 'RepVGGBlock':
                fuse_repvgg_block(model)
            if add_block == 'ACBlock':
                fuse_acblock(model)
            if add_block == 'DiverseBranchBlock':
                fuse_dbblock(model)
    else:
        if verbose:
            print('No fusion operation is required')


def save_model(model, output_dir, verbose):
    checkpoint = CheckPointer(model, save_dir=output_dir, save_to_disk=True)
    name = 'model_fused'
    checkpoint.save(name)

    model_path = os.path.join(output_dir, f'{name}.pth')
    if verbose:
        print(f'save to {model_path}')


if __name__ == '__main__':
    args = parse_args()

    cfg.merge_from_file(args.config_file)
    model = build_recognizer(cfg, device=torch.device('cpu'))

    verbose = args.verbose
    if verbose:
        print('before fuse')
        print(model)

    fuse(cfg, model, verbose)
    if verbose:
        print('after fuse')
        print(model)

    save_model(model, cfg.OUTPUT_DIR, verbose)

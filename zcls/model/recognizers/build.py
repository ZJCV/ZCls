# -*- coding: utf-8 -*-

"""
@date: 2020/11/4 下午1:58
@file: build.py
@author: zj
@description: 
"""

from torch.nn.parallel import DistributedDataParallel as DDP

from zcls.model.norm_helper import convert_sync_bn
import zcls.util.distributed as du
from zcls.util.checkpoint import CheckPointer
from zcls.util import logging

from .. import registry
from .resnet_recognizer import build_resnet
from .resnet_d_recognizer import build_resnet_d
from .mobilenetv1_recognizer import build_mobilenet_v1
from .mobilenetv2_recognizer import build_mobilenet_v2
from .resnet3d_recognizer import build_resnet3d
from .attention_resnet_recognizer import build_attention_resnet
from .shufflenetv1_recognizer import build_shufflenet_v1
from .shufflenetv2_recognizer import build_shufflenet_v2
from .mnasnet_recognizer import build_mnasnet

logger = logging.get_logger(__name__)


def build_recognizer(cfg, device):
    world_size = du.get_world_size()

    model = registry.RECOGNIZER[cfg.MODEL.RECOGNIZER.TYPE](cfg)

    if cfg.MODEL.NORM.SYNC_BN and world_size > 1:
        logger.info("start sync BN on the process group of {}".format(du.LOCAL_RANK_GROUP))
        convert_sync_bn(model, du.LOCAL_PROCESS_GROUP)
    preloaded = cfg.MODEL.RECOGNIZER.PRELOADED
    if preloaded != "":
        logger.info(f'load pretrained: {preloaded}')
        check_pointer = CheckPointer(model)
        check_pointer.load(preloaded, map_location=device)
        logger.info("finish loading model weights")

    model = model.to(device=device)
    if du.get_world_size() > 1:
        model = DDP(model, device_ids=[device], output_device=device, find_unused_parameters=True)

    return model

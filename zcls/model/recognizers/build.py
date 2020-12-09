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
from ..non_local_helper import make_non_local_2d
from .resnet_recognizer import build_resnet
from .resnet3d_recognizer import build_resnet3d
from .mobilenetv1_recognizer import build_mobilenetv1
from .mobilenetv2_recognizer import build_mobilenetv2

logger = logging.get_logger(__name__)


def build_recognizer(cfg, device):
    world_size = du.get_world_size()

    model = registry.RECOGNIZER[cfg.MODEL.NAME](cfg)

    if cfg.MODEL.NORM.SYNC_BN and world_size > 1:
        logger.info(
            "start sync BN on the process group of {}".format(du._LOCAL_RANK_GROUP))
        convert_sync_bn(model, du._LOCAL_PROCESS_GROUP)
    if cfg.MODEL.NON_LOCAL.WITH_NL:
        net_type = cfg.MODEL.RECOGNIZER.NAME
        arch_type = cfg.MODEL.BACKBONE.ARCH
        nl_type = cfg.MODEL.NON_LOCAL.NL_TYPE
        make_non_local_2d(model, net_type=net_type, arch_type=arch_type, nl_type=nl_type)
    if cfg.MODEL.PRETRAINED != "":
        logger.info(f'load pretrained: {cfg.MODEL.PRETRAINED}')
        checkpointer = CheckPointer(model)
        checkpointer.load(cfg.MODEL.PRETRAINED, map_location=device)
        logger.info("finish loading model weights")

    model = model.to(device=device)
    if du.get_world_size() > 1:
        model = DDP(model, device_ids=[device], output_device=device, find_unused_parameters=True)

    return model

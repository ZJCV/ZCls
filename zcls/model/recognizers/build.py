# -*- coding: utf-8 -*-

"""
@date: 2020/11/4 下午1:58
@file: build.py
@author: zj
@description: 
"""

from torch.nn.parallel import DistributedDataParallel as DDP

from zcls.model.conv_helper import insert_acblock, insert_repvgg_block
from zcls.model.norm_helper import convert_sync_bn
import zcls.util.distributed as du
from zcls.util.checkpoint import CheckPointer
from zcls.util import logging

from .. import registry
from .shufflenet.shufflenetv1 import ShuffleNetV1
from .shufflenet.shufflenetv2 import ShuffleNetV2
from .shufflenet.torchvision_sfv2 import build_torchvision_sfv2
from .mobilenet.mobilenetv1 import MobileNetV1
from .mobilenet.mobilenetv2 import MobileNetV2
from .mobilenet.torchvision_mobilenetv2 import build_torchvision_mbv2
from .mobilenet.mnasnet import MNASNet
from .mobilenet.torchvision_mnasnet import build_torchvision_mnasnet
from .mobilenet.mobilenetv3 import MobileNetV3
from .mobilenet.torchvision_mobilenetv3 import build_torchvision_mbv3
from .vgg.repvgg import RepVGG
from .resnet.resnet import ResNet
from .resnet.torchvision_resnet import build_torchvision_resnet
from .resnet.official_resnest import build_official_resnest
from .resnet.resnet3d import ResNet3D
from .ghostnet.ghostnet import GhostNet

logger = logging.get_logger(__name__)


def build_recognizer(cfg, device):
    world_size = du.get_world_size()

    model = registry.RECOGNIZER[cfg.MODEL.RECOGNIZER.NAME](cfg)

    if cfg.MODEL.NORM.SYNC_BN and world_size > 1:
        logger.info("start sync BN on the process group of {}".format(du.LOCAL_RANK_GROUP))
        convert_sync_bn(model, du.LOCAL_PROCESS_GROUP)
    if cfg.MODEL.CONV.ADD_BLOCKS is not None:
        assert isinstance(cfg.MODEL.CONV.ADD_BLOCKS, tuple)
        for add_block in cfg.MODEL.CONV.ADD_BLOCKS:
            if add_block == 'RepVGGBlock':
                insert_repvgg_block(model)
            if add_block == 'ACBlock':
                insert_acblock(model)
    preloaded = cfg.MODEL.RECOGNIZER.PRELOADED
    if preloaded != "":
        logger.info(f'load preloaded: {preloaded}')
        check_pointer = CheckPointer(model)
        check_pointer.load(preloaded, map_location=device)
        logger.info("finish loading model weights")

    model = model.to(device=device)
    if du.get_world_size() > 1:
        model = DDP(model, device_ids=[device], output_device=device, find_unused_parameters=False)

    return model

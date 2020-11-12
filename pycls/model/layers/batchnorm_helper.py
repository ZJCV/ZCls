# -*- coding: utf-8 -*-

"""
@date: 2020/9/23 下午2:35
@file: batchnorm_helper.py
@author: zj
@description: 
"""

import torch.nn as nn


def convert_sync_bn(model, process_group, device):
    # convert all BN layers in the model to syncBN
    for _, (child_name, child) in enumerate(model.named_children()):
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            m = nn.SyncBatchNorm.convert_sync_batchnorm(child, process_group)
            m = m.to(device=device)
            setattr(model, child_name, m)
        else:
            convert_sync_bn(child, process_group, device)

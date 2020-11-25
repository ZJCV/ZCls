# -*- coding: utf-8 -*-

"""
@date: 2020/8/29 上午10:41
@file: __init__.py.py
@author: zj
@description: 
"""

from .defaults import _C
from . import dataloader
from . import dataset
from . import lr_scheduler
from . import model
from . import optimizer
from . import transform
from . import custom_config

dataloader.add_config(_C)
dataset.add_config(_C)
lr_scheduler.add_config(_C)
model.add_config(_C)
optimizer.add_config(_C)
transform.add_config(_C)

# Add custom config with default values.
custom_config.add_custom_config(_C)


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


cfg = get_cfg_defaults()

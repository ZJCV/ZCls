# -*- coding: utf-8 -*-

"""
@date: 2021/2/2 下午5:19
@file: repvgg.py
@author: zj
@description: RegVGG，参考[RepVGG/repvgg.py](https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py)
"""

from zcls.model import registry
from zcls.model.misc import load_pretrained_weights
from ..base_recognizer import BaseRecognizer

url_map = {
    'repvgg_a0_train': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/repvgg_a0_train_imagenet_2063bcbc.pth",
    'repvgg_a0_infer': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/repvgg_a0_infer_imagenet_7af195b9.pth",
    'repvgg_a1_train': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/repvgg_a1_train_imagenet_c3def5af.pth",
    'repvgg_a1_infer': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/repvgg_a1_infer_imagenet_1b9ae0cc.pth",
    'repvgg_a2_train': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/repvgg_a2_train_imagenet_62d83c2a.pth",
    'repvgg_a2_infer': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/repvgg_a2_infer_imagenet_6f627938.pth",
    'repvgg_b0_train': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/repvgg_b0_train_imagenet_69920b69.pth",
    'repvgg_b0_infer': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/repvgg_b0_infer_imagenet_d6d57b2d.pth",
    'repvgg_b1_train': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/repvgg_b1_train_imagenet_ea3e69be.pth",
    'repvgg_b1_infer': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/repvgg_b1_infer_imagenet_64bf1d63.pth",
    'repvgg_b1g2_train': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/repvgg_b1g2_train_imagenet_902d9fee.pth",
    'repvgg_b1g2_infer': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/repvgg_b1g2_infer_imagenet_66b82aab.pth",
    'repvgg_b1g4_train': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/repvgg_b1g4_train_imagenet_5d492d86.pth",
    'repvgg_b1g4_infer': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/repvgg_b1g4_infer_imagenet_405181ae.pth",
    'repvgg_b2_train': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/repvgg_b2_train_imagenet_7c25ed20.pth",
    'repvgg_b2_infer': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/repvgg_b2_infer_imagenet_9653c204.pth",
    'repvgg_b2g4_train': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/repvgg_b2g4_train_imagenet_d4e83c66.pth",
    'repvgg_b2g4_infer': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/repvgg_b2g4_infer_imagenet_da2eaa1e.pth",
    'repvgg_b3_train': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/repvgg_b3_train_imagenet_3e026a56.pth",
    'repvgg_b3_infer': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/repvgg_b3_infer_imagenet_02e67a97.pth",
    'repvgg_b3g4_train': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/repvgg_b3g4_train_imagenet_4f7773c1.pth",
    'repvgg_b3g4_infer': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/repvgg_b3g4_infer_imagenet_3719b8ab.pth",
    'repvgg_d2se_train': "https://github.com/ZJCV/ZCls/releases/download/v0.14.0/repvgg_b3g4_train_imagenet_4f7773c1.pth",
    'repvgg_d2se_infer': "",
}


@registry.RECOGNIZER.register('RepVGG')
class RepVGG(BaseRecognizer):

    def __init__(self, cfg):
        super().__init__(cfg)

    def _init_weights(self, cfg):
        pretrained_local = cfg.MODEL.RECOGNIZER.PRETRAINED_LOCAL
        pretrained_num_classes = cfg.MODEL.RECOGNIZER.PRETRAINED_NUM_CLASSES
        num_classes = cfg.MODEL.HEAD.NUM_CLASSES

        model_name = cfg.MODEL.BACKBONE.ARCH
        assert isinstance(model_name, str)
        if model_name.startswith('repvgg'):
            if len(cfg.MODEL.CONV.ADD_BLOCKS) == 0:
                model_name += '_infer'
            else:
                model_name += '_train'
        load_pretrained_weights(self, model_name,
                                weights_path=None if pretrained_local == '' else pretrained_local,
                                load_fc=pretrained_num_classes == num_classes,
                                verbose=True,
                                url_map=url_map if cfg.MODEL.RECOGNIZER.PRETRAINED_REMOTE else None
                                )
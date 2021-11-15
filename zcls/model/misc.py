# -*- coding: utf-8 -*-

"""
@date: 2021/11/7 下午2:22
@file: misc.py
@author: zj
@description: 
"""

import os
import torch

from torch.utils import model_zoo


def load_pretrained_weights(model, model_name, weights_path=None, load_fc=True, verbose=True, url_map=None):
    """Loads pretrained weights from weights path or download using url.

    Args:
        model (Module): The whole model.
        model_name (str): Model name.
        weights_path (None or str):
            str: path to pretrained weights file on the local disk.
            None: use pretrained weights downloaded from the Internet.
        load_fc (bool): Whether to load pretrained weights for fc layer at the end of the model.
        url_map (dict or None): Remote pre-training model path corresponding to each model name
    """
    assert url_map is None or isinstance(url_map, dict), url_map
    assert weights_path is None or os.path.isfile(weights_path), weights_path

    if isinstance(weights_path, str):
        state_dict = torch.load(weights_path)['model']
    elif url_map is not None:
        if model_name not in url_map.keys() or url_map[model_name] == "":
            if verbose:
                print('No pretrained weights for {}'.format(model_name))
            return None
        remote_url = url_map[model_name]
        state_dict = model_zoo.load_url(remote_url)['model']
    else:
        return None

    if load_fc:
        ret = model.load_state_dict(state_dict, strict=False)
        assert not ret.missing_keys, 'Missing keys when loading pretrained weights: {}'.format(ret.missing_keys)
    else:
        missing_keys_list = list()
        if 'head.fc.weight' in state_dict.keys():
            state_dict.pop('head.fc.weight')
            missing_keys_list.append('head.fc.weight')
        if 'head.fc.bias' in state_dict.keys():
            state_dict.pop('head.fc.bias')
            missing_keys_list.append('head.fc.bias')
        if 'head.conv2.weight' in state_dict.keys():
            state_dict.pop('head.conv2.weight')
        if 'head.conv2.bias' in state_dict.keys():
            state_dict.pop('head.conv2.bias')
        ret = model.load_state_dict(state_dict, strict=False)
        assert set(ret.missing_keys) == set(missing_keys_list), \
            'Missing keys when loading pretrained weights: {}'.format(ret.missing_keys)
    # assert not ret.unexpected_keys, 'Missing keys when loading pretrained weights: {}'.format(ret.unexpected_keys)

    if verbose:
        print('Loaded pretrained weights for {}'.format(model_name))

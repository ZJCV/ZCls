# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午1:41
@file: misc.py
@author: zj
@description: 
"""


def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]
# -*- coding: utf-8 -*-

"""
@date: 2021/1/3 下午3:44
@file: resnetst_recognizer.py
@author: zj
@description: 
"""

"""
from 《ResNeSt: Split-Attention Networks》 Appendix
1. depth-wise convolution is not optimal for training and inference efficiency on GPU;
2. model accuracy get saturated on ImageNet with a fixed input image size;
3. increasing input image size can get better accuracy and FLOPS trade-off;
4. bicubic upsampling strategy is needed for large crop-size (≥ 320).
"""
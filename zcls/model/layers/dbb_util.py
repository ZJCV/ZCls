# -*- coding: utf-8 -*-

"""
@date: 2021/7/28 下午5:50
@file: ddb_util.py
@author: zj
@description: 
"""

from .dbb_transforms import transI_fusebn, transII_addbranch, transIII_1x1_kxk, \
    transVI_multiscale, transV_avg, transVI_multiscale

from .diverse_branch_block import DiverseBranchBlock


def get_equivalent_kernel_bias(module):
    assert isinstance(module, DiverseBranchBlock)

    k_origin, b_origin = transI_fusebn(module.dbb_origin.conv.weight, module.dbb_origin.bn)

    if hasattr(module, 'dbb_1x1'):
        k_1x1, b_1x1 = transI_fusebn(module.dbb_1x1.conv.weight, module.dbb_1x1.bn)
        k_1x1 = transVI_multiscale(k_1x1, module.kernel_size)
    else:
        k_1x1, b_1x1 = 0, 0

    if hasattr(module.dbb_1x1_kxk, 'idconv1'):
        k_1x1_kxk_first = module.dbb_1x1_kxk.idconv1.get_actual_kernel()
    else:
        k_1x1_kxk_first = module.dbb_1x1_kxk.conv1.weight
    k_1x1_kxk_first, b_1x1_kxk_first = transI_fusebn(k_1x1_kxk_first, module.dbb_1x1_kxk.bn1)
    k_1x1_kxk_second, b_1x1_kxk_second = transI_fusebn(module.dbb_1x1_kxk.conv2.weight, module.dbb_1x1_kxk.bn2)
    k_1x1_kxk_merged, b_1x1_kxk_merged = transIII_1x1_kxk(k_1x1_kxk_first, b_1x1_kxk_first, k_1x1_kxk_second,
                                                          b_1x1_kxk_second, groups=module.groups)

    k_avg = transV_avg(module.out_channels, module.kernel_size, module.groups)
    k_1x1_avg_second, b_1x1_avg_second = transI_fusebn(k_avg.to(module.dbb_avg.avgbn.weight.device),
                                                       module.dbb_avg.avgbn)
    if hasattr(module.dbb_avg, 'conv'):
        k_1x1_avg_first, b_1x1_avg_first = transI_fusebn(module.dbb_avg.conv.weight, module.dbb_avg.bn)
        k_1x1_avg_merged, b_1x1_avg_merged = transIII_1x1_kxk(k_1x1_avg_first, b_1x1_avg_first, k_1x1_avg_second,
                                                              b_1x1_avg_second, groups=module.groups)
    else:
        k_1x1_avg_merged, b_1x1_avg_merged = k_1x1_avg_second, b_1x1_avg_second

    return transII_addbranch((k_origin, k_1x1, k_1x1_kxk_merged, k_1x1_avg_merged),
                             (b_origin, b_1x1, b_1x1_kxk_merged, b_1x1_avg_merged))

# -*- coding: utf-8 -*-

"""
@date: 2021/10/3 下午3:06
@file: dropout.py
@author: zj
@description: 
"""

import cv2


from zcls.data.transforms.realization import CoarseDropout


def dropout():
    image = cv2.imread('tests/assets/lena_224.jpg')

    m = CoarseDropout(max_holes=8, max_height=8, max_width=8, min_holes=None, min_height=None, min_width=None,
                      fill_value=0, p=0.5)
    print(m)
    res = m(image)
    cv2.imwrite('tests/assets/lena_dropout.jpg', res)


if __name__ == '__main__':
    dropout()

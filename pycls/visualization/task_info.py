# -*- coding: utf-8 -*-

"""
@date: 2020/10/22 上午9:20
@file: task_info.py
@author: zj
@description: 
"""


class TaskInfo:
    def __init__(self):
        self.frames = None
        self.id = -1
        self.action_preds = None
        self.num_buffer_frames = 0
        self.img_height = -1
        self.img_width = -1
        self.crop_size = -1
        self.clip_vis_size = -1

    def add_frames(self, idx, frames):
        """
        Add the clip and corresponding id.
        Args:
            idx (int): the current index of the clip.
            frames (list[ndarray]): list of images in "BGR" format.
        """
        self.frames = frames
        self.id = idx

    def add_action_preds(self, preds):
        """
        Add the corresponding action predictions.
        """
        self.action_preds = preds

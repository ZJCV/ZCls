# -*- coding: utf-8 -*-

"""
@date: 2020/10/22 上午9:29
@file: action_predictor.py
@author: zj
@description: 
"""

import queue

from .predictor import Predictor


class ActionPredictor:
    """
    Synchronous Action Prediction and Visualization pipeline with AsyncVis.
    """

    def __init__(self, cfg, async_vis=None):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                pycls/config/defaults.py
            async_vis (AsyncVis object): asynchronous visualizer.
        """
        self.predictor = Predictor(cfg=cfg)
        self.async_vis = async_vis

    def put(self, task):
        """
        Make prediction and put the results in `async_vis` task queue.
        Args:
            task (TaskInfo object): task object that contain
                the necessary information for action prediction. (e.g. frames, boxes)
        """
        task = self.predictor(task)
        self.async_vis.get_indices_ls.append(task.id)
        self.async_vis.put(task)

    def get(self):
        """
        Get the visualized clips if any.
        """
        try:
            task = self.async_vis.get()
        except (queue.Empty, IndexError):
            raise IndexError("Results are not available yet.")

        return task

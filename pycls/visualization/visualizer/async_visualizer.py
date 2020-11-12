# -*- coding: utf-8 -*-

"""
@date: 2020/10/22 上午9:37
@file: async_visualizer.py
@author: zj
@description: 
"""

import atexit
import numpy as np
import torch.multiprocessing as mp

from pycls.visualization.stop_token import _StopToken
from .util import draw_predictions
from .video_visualizer import VideoVisualizer


class AsyncVisualizer:
    class _VisWorker(mp.Process):
        def __init__(self, video_vis, task_queue, result_queue):
            """
            Visualization Worker for AsyncVis.
            Args:
                video_vis (VideoVisualizer object): object with tools for visualization.
                task_queue (mp.Queue): a shared queue for incoming task for visualization.
                result_queue (mp.Queue): a shared queue for visualized results.
            """
            self.video_vis = video_vis
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            """
            Run visualization asynchronously.
            """
            while True:
                task = self.task_queue.get()
                if isinstance(task, _StopToken):
                    break

                frames = draw_predictions(task, self.video_vis)
                task.frames = np.array(frames)
                self.result_queue.put(task)

    def __init__(self, cfg, n_workers=None):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                pycls/config/defaults.py
            n_workers (Optional[int]): number of CPUs for running video visualizer.
                If not given, use all CPUs.
        """

        num_workers = mp.cpu_count() if n_workers is None else n_workers

        common_classes = (
            cfg.VISUALIZATION.COMMON_CLASS_NAMES
            if len(cfg.VISUALIZATION.LABEL_FILE_PATH) != 0
            else None
        )
        video_vis = VideoVisualizer(
            num_classes=cfg.MODEL.HEAD.NUM_CLASSES,
            class_names_path=cfg.VISUALIZATION.LABEL_FILE_PATH,
            colormap=cfg.VISUALIZATION.COLORMAP,
            thres=cfg.VISUALIZATION.COMMON_CLASS_THRES,
            lower_thres=cfg.VISUALIZATION.UNCOMMON_CLASS_THRES,
            common_class_names=common_classes,
            mode=cfg.VISUALIZATION.VIS_MODE
        )

        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.get_indices_ls = []
        self.procs = []
        self.result_data = {}
        self.put_id = -1
        for _ in range(max(num_workers, 1)):
            self.procs.append(
                AsyncVisualizer._VisWorker(
                    video_vis, self.task_queue, self.result_queue
                )
            )

        for p in self.procs:
            p.start()

        atexit.register(self.shutdown)

    def put(self, task):
        """
        Add the new task to task queue.
        Args:
            task (TaskInfo object): task object that contain
                the necessary information for action prediction. (e.g. frames, boxes, predictions)
        """
        self.put_id += 1
        self.task_queue.put(task)

    def get(self):
        """
        Return visualized frames/clips in the correct order based on task id if
        result(s) is available. Otherwise, raise queue.Empty exception.
        """
        get_idx = self.get_indices_ls[0]
        if self.result_data.get(get_idx) is not None:
            res = self.result_data[get_idx]
            del self.result_data[get_idx]
            del self.get_indices_ls[0]
            return res

        while True:
            res = self.result_queue.get(block=False)
            idx = res.id
            if idx == get_idx:
                del self.get_indices_ls[0]
                return res
            self.result_data[idx] = res

    def __call__(self, task):
        """
        How many results are ready to be returned.
        """
        self.put(task)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(_StopToken())

    @property
    def result_available(self):
        return self.result_queue.qsize() + len(self.result_data)

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5

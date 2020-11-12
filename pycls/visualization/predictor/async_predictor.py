#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import atexit
import torch
import torch.multiprocessing as mp

from .predictor import Predictor
from pycls.visualization.stop_token import _StopToken


class AsyncPredictor:
    class _Predictor(mp.Process):
        def __init__(self, cfg, task_queue, result_queue, gpu_id=None):
            """
            Predict Worker for Detectron2.
            Args:
                cfg (CfgNode): configs. Details can be found in
                    pycls/config/defaults.py
                task_queue (mp.Queue): a shared queue for incoming task.
                result_queue (mp.Queue): a shared queue for predicted results.
                gpu_id (int): index of the GPU device for the current child process.
            """
            super().__init__()
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            self.gpu_id = gpu_id

            self.device = (
                torch.device("cuda:{}".format(self.gpu_id))
                if self.cfg.NUM_GPUS
                else "cpu"
            )

        def run(self):
            """
            Run prediction asynchronously.
            """
            # Build the video model and print model statistics.
            model = Predictor(self.cfg, gpu_id=self.gpu_id)
            while True:
                task = self.task_queue.get()
                if isinstance(task, _StopToken):
                    break
                task = model(task)
                self.result_queue.put(task)

    def __init__(self, cfg, result_queue=None):
        num_workers = cfg.NUM_GPUS

        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue() if result_queue is None else result_queue

        self.get_idx = -1
        self.put_idx = -1
        self.procs = []
        cfg = cfg.clone()
        cfg.defrost()
        cfg.NUM_GPUS = 1
        for gpu_id in range(num_workers):
            self.procs.append(
                AsyncPredictor._Predictor(
                    cfg, self.task_queue, self.result_queue, gpu_id
                )
            )

        self.result_data = {}
        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, task):
        """
        Add the new task to task queue.
        Args:
            task (TaskInfo object): task object that contain
                the necessary information for action prediction. (e.g. frames)
        """
        self.put_idx += 1
        self.task_queue.put(task)

    def get(self):
        """
        Return a task object in the correct order based on task id if
        result(s) is available. Otherwise, raise queue.Empty exception.
        """
        if self.result_data.get(self.get_idx + 1) is not None:
            self.get_idx += 1
            res = self.result_data[self.get_idx]
            del self.result_data[self.get_idx]
            return res
        while True:
            res = self.result_queue.get(block=False)
            idx = res.id
            if idx == self.get_idx + 1:
                self.get_idx += 1
                return res
            self.result_data[idx] = res

    def __call__(self, task):
        self.put(task)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(_StopToken())

    @property
    def result_available(self):
        """
        How many results are ready to be returned.
        """
        return self.result_queue.qsize() + len(self.result_data)

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5

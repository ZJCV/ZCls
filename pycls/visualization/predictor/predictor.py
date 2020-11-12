#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch

from pycls.model.recognizers.build import build_recognizer
from pycls.data.transforms.build import build_transform
from .util import process_cv2_inputs
from pycls.util.distributed import get_device, get_local_rank


class Predictor:
    """
    Action Predictor for action recognition.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                pycls/config/defaults.py
            gpu_id (Optional[int]): GPU id.
        """
        if cfg.NUM_GPUS > 0:
            device = get_device(local_rank=get_local_rank())
        else:
            device = get_device()

        # Build the video model and print model statistics.
        self.model = build_recognizer(cfg, device)
        self.model.eval()
        self.transform = build_transform(cfg, is_train=False)

        self.cfg = cfg
        self.device = device

    def __call__(self, task):
        """
        Returns the prediction results for the current task.
        Args:
            task (TaskInfo object): task object that contain
                the necessary information for action prediction. (e.g. frames)
        Returns:
            task (TaskInfo object): the same task info object but filled with
                prediction values (a tensor)
        """
        frames = task.frames

        inputs = process_cv2_inputs(frames, self.cfg, self.transform).to(device=self.device, non_blocking=True)

        preds = self.model(inputs)['probs']
        preds = torch.softmax(preds, dim=1)

        if self.cfg.NUM_GPUS:
            preds = preds.cpu()

        preds = preds.detach()
        task.add_action_preds(preds)

        return task

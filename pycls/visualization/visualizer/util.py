# -*- coding: utf-8 -*-

"""
@date: 2020/10/22 上午9:25
@file: util.py
@author: zj
@description: 
"""

import os
import json

import pycls.util.logging as logging


def get_class_names(path):
    """
    Read json file with entries {classname: index} and return
    an array of class names in order.
    Args:
        path (str): path to class ids json file.
            File must be in the format {"class1": id1, "class2": id2, ...}
    Returns:
        class_names (list of strs): list of class names.
    """
    assert os.path.exists(path), f'{path} is None'
    # try:
    #     with open(path, "r") as f:
    #         class2idx = json.load(f)
    # except Exception as err:
    #     print("Fail to load file from {} with error {}".format(path, err))
    #     return
    #
    # class_names = [None] * len(class2idx)
    # # 如果类标签从1开始
    # if min(class2idx.values()) == 1:
    #     for k, i in class2idx.items():
    #         class_names[i - 1] = k
    # else:
    #     # 从0开始
    #     for k, i in class2idx.items():
    #         class_names[i] = k

    with open(path, 'r') as f:
        class_names = [line.strip().split(' ')[1] for line in f]

    return class_names


def create_text_labels(classes, scores, class_names, ground_truth=False):
    """
    Create text labels.
    Args:
        classes (list[int]): a list of class ids for each example.
        scores (list[float] or None): list of scores for each example.
        class_names (list[str]): a list of class names, ordered by their ids.
        ground_truth (bool): whether the labels are ground truth.
    Returns:
        labels (list[str]): formatted text labels.
    """
    try:
        labels = [class_names[i] for i in classes]
    except IndexError:
        logger = logging.setup_logging(__name__)
        logger.error("Class indices get out of range: {}".format(classes))
        return None

    if ground_truth:
        labels = ["[{}] {}".format("GT", label) for label in labels]
    elif scores is not None:
        assert len(classes) == len(scores)
        labels = [
            "[{:.2f}] {}".format(s, label) for s, label in zip(scores, labels)
        ]
    return labels


def draw_predictions(task, video_vis):
    """
    Draw prediction for the given task.
    Args:
        task (TaskInfo object): task object that contain
            the necessary information for visualization. (e.g. frames, preds)
            All attributes must lie on CPU devices.
        video_vis (VideoVisualizer object): the video visualizer object.
    """
    frames = task.frames
    preds = task.action_preds

    keyframe_idx = len(frames) // 2 - task.num_buffer_frames
    draw_range = [
        keyframe_idx - task.clip_vis_size,
        keyframe_idx + task.clip_vis_size,
    ]
    buffer = frames[: task.num_buffer_frames]
    frames = frames[task.num_buffer_frames:]

    # frames = video_vis.draw_clip_range(
    #     frames, preds, keyframe_idx=keyframe_idx, draw_range=draw_range
    # )
    frames = video_vis.draw_clip(frames, preds)
    del task

    return buffer + frames

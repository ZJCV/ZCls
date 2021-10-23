# -*- coding: utf-8 -*-

"""
@date: 2020/10/19 上午10:03
@file: base_evaluator.py
@author: zj
@description: 
"""

from abc import ABCMeta, abstractmethod
import torch

from zcls.config.key_word import KEY_OUTPUT
from zcls.util.metrics import topk_accuracy


class BaseEvaluator(metaclass=ABCMeta):

    def __init__(self, classes, top_k=(1,)):
        super(BaseEvaluator, self).__init__()
        self.classes = classes
        self.device = torch.device('cpu')

        self.top_k = top_k
        self._init()

    def _init(self):
        self.total_outputs_list = list()
        self.total_targets_list = list()

        self.cate_outputs_dict = dict()
        self.cate_targets_dict = dict()
        for i in range(len(self.classes)):
            key = str(i)
            self.cate_outputs_dict[key] = list()
            self.cate_targets_dict[key] = list()

    def evaluate_train(self, output_dict: dict, targets: torch.Tensor):
        assert isinstance(output_dict, dict) and KEY_OUTPUT in output_dict.keys()

        probs = output_dict[KEY_OUTPUT]
        res, _ = topk_accuracy(probs, targets, top_k=self.top_k)

        acc_dict = dict()
        for i in range(len(self.top_k)):
            acc_dict[f'tok{self.top_k[i]}'] = res[i]
        return acc_dict

    def evaluate_test(self, output_dict: dict, targets: torch.Tensor):
        assert isinstance(output_dict, dict) and KEY_OUTPUT in output_dict.keys()
        probs = output_dict[KEY_OUTPUT]
        outputs = probs.to(device=self.device)
        targets = targets.to(device=self.device)

        self.total_outputs_list.extend(outputs)
        self.total_targets_list.extend(targets)

        for i, target in enumerate(targets):
            key = str(target.item())
            self.cate_outputs_dict[key].append(outputs[i])
            self.cate_targets_dict[key].append(targets[i])

    def get(self):
        assert len(self.total_targets_list) == len(self.total_outputs_list)
        if len(self.total_targets_list) == 0:
            return None, None

        result_str = '\ntotal({}) -'.format(len(self.total_targets_list))
        topk_list, correct_topk_list = topk_accuracy(torch.stack(self.total_outputs_list),
                                                     torch.stack(self.total_targets_list),
                                                     top_k=self.top_k)
        acc_dict = dict()
        for i in range(len(self.top_k)):
            acc_dict[f"top{self.top_k[i]}"] = topk_list[i]
            result_str += ' top{} acc({}): {:6.3f}'.format(self.top_k[i], correct_topk_list[i], topk_list[i])

        for idx in range(len(self.classes)):
            class_name = self.classes[idx].strip()

            key = str(idx)
            cate_outputs = self.cate_outputs_dict[key]
            result_str += '\n{:<3}\t- {:<20}\t'.format(idx, class_name)
            if len(cate_outputs) == 0:
                for i in range(len(self.top_k)):
                    acc_dict[f"top{self.top_k[i]}"] = 0.
                    result_str += '\ttop{}(0): {:<5}'.format(self.top_k[i], "{:.2f}".format(0.))
            else:
                cate_outputs = torch.stack(self.cate_outputs_dict[key])
                cate_targets = torch.stack(self.cate_targets_dict[key])

                topk_list, correct_topk_list = topk_accuracy(cate_outputs, cate_targets, top_k=self.top_k)

                for i in range(len(self.top_k)):
                    acc_dict[f"top{self.top_k[i]}"] = topk_list[i]
                    result_str += '\ttop{}({}): {:<5}'.format(self.top_k[i],
                                                              correct_topk_list[i],
                                                              "{:.2f}".format(topk_list[i]))
        result_str += '\n'

        return result_str, acc_dict

    def clean(self):
        self._init()

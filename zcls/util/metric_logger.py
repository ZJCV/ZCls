from collections import deque, defaultdict
import numpy as np
import datetime
import torch

from .distributed import all_reduce
from .misc import gpu_mem_usage, cpu_mem_usage


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=10):
        self.deque = deque(maxlen=window_size)
        self.value = np.nan
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value
        self.value = value

    @property
    def median(self):
        values = np.array(self.deque)
        return np.median(values)

    @property
    def avg(self):
        values = np.array(self.deque)
        return np.mean(values)

    @property
    def global_avg(self):
        return self.total / self.count


class MetricLogger:
    def __init__(self, delimiter=", "):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            if 'loss' in name:
                loss_str.append(
                    "{}: {:.6f} ({:.6f})".format(name, meter.avg, meter.global_avg)
                )
            else:
                loss_str.append(
                    "{}: {:.3f} ({:.3f})".format(name, meter.avg, meter.global_avg)
                )
        return self.delimiter.join(loss_str)


def update_stats(num_gpus, meters, loss_dict, acc_dict):
    assert isinstance(loss_dict, dict) and isinstance(acc_dict, dict)

    # Gather all the predictions across all the devices.
    keys = list()
    values = list()
    for key in sorted(loss_dict.keys()):
        keys.append(key)
        values.append(loss_dict[key])
    for key in sorted(acc_dict.keys()):
        keys.append(key)
        values.append(acc_dict[key])
    if num_gpus > 1:
        reduced_values = all_reduce(values)
        meter_dict = {k: v for k, v in zip(keys, reduced_values)}

        meters.update(**meter_dict)
    else:
        meter_dict = {k: v for k, v in zip(keys, values)}
        meters.update(**meter_dict)


def log_iter_stats(cur_iter, epoch_iters, cur_epoch, max_epoch, lr, meters):
    max_iter = epoch_iters * max_epoch
    cur_epoch_iter = epoch_iters * (cur_epoch - 1) + (cur_iter + 1)
    eta_seconds = meters.time.global_avg * (max_iter - cur_epoch_iter)
    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

    stats = meters.delimiter.join([
        "epoch: {cur_epoch:04d}/{max_epoch:04d}",
        "iter: {cur_iter:05d}/{max_iter:05d}",
        "lr: {lr:.5f}",
        '{meters}',
        "eta: {eta}",
        'gpu_mem: {mem:.2f}G',
    ]).format(
        cur_epoch=cur_epoch,
        max_epoch=max_epoch,
        cur_iter=cur_iter + 1,
        max_iter=epoch_iters,
        lr=lr,
        meters=str(meters),
        eta=eta_string,
        mem=gpu_mem_usage(),
    )

    return stats


def log_epoch_stats(epoch_iters, cur_epoch, max_epoch, lr, meters):
    max_iter = epoch_iters * max_epoch
    cur_epoch_iter = epoch_iters * cur_epoch
    eta_seconds = meters.time.global_avg * (max_iter - cur_epoch_iter)
    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

    usage, total = cpu_mem_usage()

    stats = meters.delimiter.join([
        "epoch: {cur_epoch:04d}/{max_epoch:04d}",
        "lr: {lr:.5f}",
        "eta: {eta}",
        'gpu_mem: {mem:.2f}G',
        "RAM: {usage:.2f}/{total:.2f}G",
    ]).format(
        cur_epoch=cur_epoch,
        max_epoch=max_epoch,
        lr=lr,
        meters=str(meters),
        eta=eta_string,
        mem=gpu_mem_usage(),
        usage=usage,
        total=total,
    )

    return stats

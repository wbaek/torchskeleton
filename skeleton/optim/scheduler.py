# -*- coding: utf-8 -*-
from __future__ import absolute_import
import math
import logging

import numpy as np


LOGGER = logging.getLogger(__name__)


def gradual_warm_up(scheduler, warm_up_epoch, multiplier):
    def schedule(e, **kwargs):
        lr = scheduler(e, **kwargs)
        lr = lr * ((multiplier - 1.0) * min(e, warm_up_epoch) / warm_up_epoch + 1)
        return lr
    return schedule


def get_discrete_epoch(scheduler):
    def schedule(e, **kwargs):
        return scheduler(int(e), **kwargs)
    return schedule


def get_change_scale(scheduler, init_scale=1.0):
    def schedule(e, scale=None, **kwargs):
        lr = scheduler(e, **kwargs)
        return lr * (scale if scale is not None else init_scale)
    return schedule


def get_piecewise(knots, vals):
    def schedule(e, **kwargs):
        return np.interp([e], knots, vals)[0]
    return schedule


def get_step_scheduler(init_lr, step_size, gamma=0.1, scheduler=None):
    def schedule(e, **kwargs):
        lr = init_lr
        if scheduler is not None:
            lr = scheduler(e, **kwargs)
        lr = lr * gamma ** int(e / step_size)
        return lr
    return schedule


def get_cosine_scheduler(init_lr, maximum_epoch, eta_min=0, scheduler=None):
    def schedule(e, **kwargs):
        lr = init_lr
        if scheduler is not None:
            lr = scheduler(e, **kwargs)
        maximum = kwargs['maximum_epoch'] if 'maximum_epoch' in kwargs else maximum_epoch
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * e / maximum)) / 2
        return lr
    return schedule


def get_reduce_on_plateau_scheduler(init_lr, factor=0.1, patience=10, threshold=1e-4, min_lr=0, metric_name='metric'):
    class Schedule:
        def __init__(self):
            self.num_bad_epochs = 0
            self.lr = init_lr
            self.best = None
            self.metric_name = metric_name

        def __call__(self, e, **kwargs):
            if self.metric_name not in kwargs:
                return self.lr
            metric = kwargs[self.metric_name]

            LOGGER.debug(
                '[%s] lr:%f best:%f curr:%f num_bad_epoch:%d>%d',
                'get_reduce_on_plateau',
                self.lr,
                self.best if self.best is not None else -1,
                metric,
                self.num_bad_epochs,
                patience
            )

            if self.best is None or self.best > metric:
                self.best = metric - threshold
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.num_bad_epochs > patience:
                self.num_bad_epochs = 0
                lr = max(min_lr, self.lr * factor)
                LOGGER.debug('[%s] reduce lr %f -> %f', 'get_reduce_on_plateau', self.lr, lr)
                self.lr = lr
            return self.lr
    return Schedule()

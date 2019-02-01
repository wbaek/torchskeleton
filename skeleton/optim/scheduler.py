# -*- coding: utf-8 -*-
from __future__ import absolute_import
import math
import logging


LOGGER = logging.getLogger(__name__)


def gradual_warm_up(scheduler, maximum_epoch, multiplier):
    def schedule(e):
        lr = scheduler(int(e))
        lr = lr * ((multiplier - 1.0) * min(e, maximum_epoch) / maximum_epoch + 1)
        return lr
    return schedule


def get_discrete_epoch(scheduler):
    def schedule(e):
        return scheduler(int(e))
    return schedule


def get_cosine_scheduler(init_lr, maximum_epoch, eta_min=0):
    def schedule(e):
        lr = eta_min + (init_lr - eta_min) * (1 + math.cos(math.pi * e / maximum_epoch)) / 2
        return lr
    return schedule


def get_step_scheduler(init_lr, step_size, gamma=0.1):
    def schedule(e):
        lr = init_lr * gamma ** (e // step_size)
        return lr
    return schedule

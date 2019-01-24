# -*- coding: utf-8 -*-
from __future__ import absolute_import
import logging

import torch


LOGGER = logging.getLogger(__name__)


class ScheduledOptimzer:
    def __init__(self, parameters, optimizer, steps_per_epoch=1, clip_grad_max_norm=5.0, **opt_params):
        self.epoch = 0.0
        self.steps_per_epoch = steps_per_epoch
        self.clip_grad_max_norm = clip_grad_max_norm
        self.opt_params = opt_params
        self._parameters = parameters
        self._optimizer = optimizer(parameters, **self.get_opt_params())

    def get_opt_params(self):
        return {k: v(self.epoch) if callable(v) else v for k, v in self.opt_params.items()}

    def step(self, epoch=None):
        self.epoch = self.epoch + (1.0 / self.steps_per_epoch) if epoch is None else epoch

        self._optimizer.param_groups[0].update(**self.get_opt_params())
        torch.nn.utils.clip_grad_norm_(self._parameters, self.clip_grad_max_norm, norm_type=2)
        self._optimizer.step()

    def state_dict(self):
        state_dict = self._optimizer.state_dict()
        state_dict.update({'epoch', self.epoch})
        return state_dict

    def load_state_dict(self, state_dict):
        self.epoch = state_dict.pop('epoch')
        return self._optimizer.load_state_dict(state_dict)

    def zero_grad(self):
        return self._optimizer.zero_grad()

    def __getattr__(self, item):
        return getattr(self._optimizer, item)

# -*- coding: utf-8 -*-
from __future__ import absolute_import
import logging

import torch
from ..summary import summary


LOGGER = logging.getLogger(__name__)


class ScheduledOptimzer:
    def __init__(self, parameters, optimizer, steps_per_epoch=1, clip_grad_max_norm=0.0, **opt_params):
        self.epoch = 0.0
        self._parameters = parameters
        self.steps_per_epoch = steps_per_epoch
        self.clip_grad_max_norm = clip_grad_max_norm
        self._opt_params = opt_params

        self.opt_params = {
            k: v(0) if callable(v) else v
            for k, v in self._opt_params.items()
        }
        self._optimizer = optimizer(parameters, **self.opt_params)

    def update(self, epoch=None):
        self.opt_params = {
            k: v(self.epoch if epoch is None else epoch) if callable(v) else v
            for k, v in self._opt_params.items()
        }
        self._optimizer.param_groups[0].update(**self.opt_params)
        for key, value in self.opt_params.items():
            summary.scalar('train', 'optimizer/%s' % key, value)
        # LOGGER.debug('update optimizer params:%s', self.opt_params)
        return self

    def step(self, epoch=None):
        self.update(self.epoch)
        self.epoch = self.epoch + (1.0 / self.steps_per_epoch) if epoch is None else epoch
        if self.clip_grad_max_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self._parameters, self.clip_grad_max_norm, norm_type=2)
        self._optimizer.step()

    def state_dict(self):
        state_dict = self._optimizer.state_dict()
        state_dict.update({'epoch': self.epoch})
        return state_dict

    def load_state_dict(self, state_dict):
        self.epoch = state_dict.pop('epoch')
        return self._optimizer.load_state_dict(state_dict)

    def zero_grad(self):
        return self._optimizer.zero_grad()

    def __getattr__(self, item):
        return getattr(self._optimizer, item)

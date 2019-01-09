# -*- coding: utf-8 -*-
from __future__ import absolute_import
import os
import copy
import logging

import torch
from torch import nn


LOGGER = logging.getLogger(__name__)


class IOModule(nn.Module):  # pylint: disable=abstract-method
    def copy(self):
        return copy.deepcopy(self)

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.state_dict(), filepath)
        return self

    def load(self, filepath):
        self.load_state_dict(torch.load(filepath))
        return self


class MoveToModule(nn.Module):
    @staticmethod
    def hook(module, inputs):
        for t in inputs:
            if not isinstance(t, torch.Tensor):
                continue
            if module.is_half:
                if t.is_floating_point():
                    t.data = t.data.half()
            if module.to_args or module.to_kwargs:
                t.data = t.data.to(*module.to_args, **module.to_kwargs)

    def __init__(self):
        super(MoveToModule, self).__init__()
        self.to_args = ()
        self.to_kwargs = {}
        self.is_half = False
        self.register_forward_pre_hook(MoveToModule.hook)

    def to(self, *args, **kwargs):
        self.to_args = args
        self.to_kwargs = kwargs
        return super(MoveToModule, self).to(*args, **kwargs)

    def half(self):
        self.is_half = True
        return super(MoveToModule, self).half()

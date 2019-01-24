# -*- coding: utf-8 -*-
# pylint: disable=arguments-differ, abstract-method
from __future__ import absolute_import
import os
import copy
import logging

import torch
from torch import nn


LOGGER = logging.getLogger(__name__)


class IOModule(nn.Module):
    def copy(self):
        return copy.deepcopy(self)

    def save(self, filepath):
        try:
            os.makedirs(os.path.dirname(filepath))
        except OSError:
            pass  # already path exists
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
        self.to_kwargs = {'non_blocking': True}
        self.is_half = False
        self.handle = self.register_forward_pre_hook(MoveToModule.hook)

    def to(self, *args, **kwargs):
        self.to_args = args
        self.to_kwargs.update(kwargs)
        return super(MoveToModule, self).to(*args, **kwargs)

    def half(self):
        self.is_half = True
        return super(MoveToModule, self).half()

    def remove_hooks(self):
        self.handle.remove()
        return self


class VerboseModule(nn.Module):
    def __init__(self):
        super(VerboseModule, self).__init__()
        self.handles = []

    def print_architecture(self):
        for name, module in self.layers.named_modules():
            if name == '':
                continue
            split = name.split('.')
            indent = '\t' * (len(split) - 1)
            class_name = module.__class__.__name__ if not module.__class__.__name__ == 'Sequential' else ''
            print(indent, split[-1], class_name)

    def register_verbose_hooks(self):
        def verbose(module, inputs):
            input_shape = inputs[0].shape if len(inputs[0]) == 1 else [len(inputs[0])] + list(inputs[0][0].shape)
            print('%20s shape input to %s' % (list(input_shape), type(module)))

        def register(m):
            if not isinstance(m, torch.nn.Sequential):
                handle = m.register_forward_pre_hook(verbose)
                self.handles.append(handle)

        self.apply(register)
        return self

    def remove_verbose_hooks(self):
        _ = [h.remove() for h in self.handles]
        self.handles = []
        return self

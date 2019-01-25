# -*- coding: utf-8 -*-
# pylint: disable=arguments-differ, abstract-method
from __future__ import absolute_import
import logging

import torch
from torch import nn


LOGGER = logging.getLogger(__name__)


class VerboseModule(nn.Module):
    def __init__(self):
        super(VerboseModule, self).__init__()
        self.handles = []

    def print_architecture(self):
        for name, module in self.named_modules():
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

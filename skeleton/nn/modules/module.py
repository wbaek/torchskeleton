# -*- coding: utf-8 -*-
# pylint: disable=arguments-differ, abstract-method
from __future__ import absolute_import
import logging
from collections import OrderedDict

import torch
from torch import nn
from treelib import Tree, Node


LOGGER = logging.getLogger(__name__)


class TraceModule(nn.Module):
    def __init__(self):
        super(TraceModule, self).__init__()
        self.handles = []
        self.flatten_forward_pass = []

    def forward(self, *inputs):
        self.flatten_forward_pass = []
        super.__format__(*inputs)

    def print_trace(self):
        tree = Tree()
        tree._nodes = OrderedDict([])

        for node in self.flatten_forward_pass:
            name = node['name_current'] if node['name_current'] else 'root'
            identifier = node['name'] if node['name'] else 'root'

            idx = 0
            while True:
                if not tree.contains(identifier):
                    break
                idx += 1
                if not tree.contains('%s_%d' % (identifier, idx)):
                    identifier = '%s_%d' % (identifier, idx)
                    break

            if identifier == 'root':
                parent = None
            else:
                parent = node['name_parent'] if node['name_parent'] else 'root'

            if idx == 0:
                tag = '[%s] %s: %s' % (name, node['class_name'], node['input_shape'])
            else:
                tag = '[%s] (shared:%d) %s: %s' % (name, idx, node['class_name'], node['input_shape'])
            tree.create_node(tag=tag, identifier=identifier, parent=parent)

        tree.show(key=False)

    def register_trace_hooks(self):
        def get_hook(name):
            def verbose(module, inputs):
                class_name = module.__class__.__name__
                split = name.split('.')
                inputs = inputs[0]
                input_shape = inputs.shape if isinstance(inputs, torch.Tensor) else [tuple(inputs_.shape) for inputs_ in inputs]

                self.flatten_forward_pass.append({
                    'name': name,
                    'name_parent': '.'.join(split[:-1]),
                    'name_current': split[-1],
                    'class_name': class_name,
                    'input_shape': tuple(input_shape),
                })
            return verbose

        for name, module in self.named_modules():
            handle = module.register_forward_pre_hook(get_hook(name))
            self.handles.append(handle)
        return self

    def remove_trace_hooks(self):
        _ = [h.remove() for h in self.handles]
        self.handles = []
        return self

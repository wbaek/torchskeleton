# -*- coding: utf-8 -*-
# pylint: disable=arguments-differ, abstract-method
from __future__ import absolute_import
import logging
from collections import OrderedDict

import torch
from treelib import Tree


LOGGER = logging.getLogger(__name__)


class OrderedTree(Tree):
    def __init__(self):
        super(OrderedTree, self).__init__()
        self._nodes = OrderedDict([])
        self._reader = ""

    def show(self, nid=None, level=Tree.ROOT, idhidden=True, filter=None,  # pylint: disable=redefined-builtin
             key=False, reverse=False, line_type='ascii', data_property=None):
        super(OrderedTree, self).show(nid=nid, level=level, idhidden=idhidden, filter=filter,
                                      key=key, reverse=reverse, line_type=line_type, data_property=data_property)
        return self._reader

class TraceModule(torch.nn.Module):
    def __init__(self):
        super(TraceModule, self).__init__()
        self.handles = []
        self.flatten_forward_pass = []

    def print_trace(self):
        tree = OrderedTree()

        def add_nodes(tree, identifier):
            if identifier is None or not identifier:
                return
            split = identifier.split('.')
            for i in range(2, len(split)+1):
                idx = '.'.join(split[:i])
                name = split[i-1]
                if not tree.contains(idx):
                    tree.create_node(tag='[%s] None' % name, identifier=idx, parent='.'.join(split[:i-1]))

        for i, node in enumerate(self.flatten_forward_pass):
            name = node['name_current'] if node['name_current'] else 'root'
            identifier = node['name'] if node['name'] else 'root'
            if identifier == 'root':
                parent = None
            else:
                parent = node['name_parent'] if node['name_parent'] else 'root'

            idx = 0
            while True:
                if not tree.contains(identifier):
                    break
                idx += 1
                if not tree.contains('%s_%d' % (identifier, idx)):
                    identifier = '%s_%d' % (identifier, idx)
                    break

            if idx == 0:
                tag = '[%s] %s: %s' % (name, node['class_name'], node['input_shape'])
            else:
                tag = '[%s] %s: %s (shared:%s)' % (name, node['class_name'], node['input_shape'], node['name'])
                parent = self.flatten_forward_pass[i-1]['name_parent']

            add_nodes(tree, parent)
            tree.create_node(tag=tag, identifier=identifier, parent=parent)
        return tree.show()

    def register_trace_hooks(self):
        def get_hook(name):
            def verbose(module, inputs):
                if not inputs or inputs[0] is None:
                    return
                class_name = module.__class__.__name__
                split = name.split('.')
                parent = '.'.join(split[:-1])
                inputs = inputs[0]
                input_shape = inputs.shape if isinstance(inputs, torch.Tensor) else [tuple(inputs_.shape) if inputs_ is not None else (None,) for inputs_ in inputs]

                data = {
                    'name': name,
                    'name_parent': parent,
                    'name_current': split[-1],
                    'class_name': class_name,
                    'input_shape': tuple(input_shape),
                }
                self.flatten_forward_pass.append(data)
            return verbose

        for name, module in self.named_modules():
            handle = module.register_forward_pre_hook(get_hook(name))
            self.handles.append(handle)
        return self

    def remove_trace_hooks(self):
        _ = [h.remove() for h in self.handles]
        self.handles = []
        return self

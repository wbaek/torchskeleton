# -*- coding: utf-8 -*-
# pylint: disable=arguments-differ, abstract-method
from __future__ import absolute_import
import logging
from collections import OrderedDict

import torch

from .operations import ReLUConvBN, FactorizedReduce, Identity
from ..nn.modules import DropPath


LOGGER = logging.getLogger(__name__)


class Cell(torch.nn.Module):
    def __init__(self, operations, nodes, channels, in_channels, prev_channels, prev_reduce, affine=True):
        '''
        :param operations:
            operations = [
            {'to': 2, 'from': 0, 'op': torch.nn.Conv2d()},
            ...
            ]
            nodes = [2, 3, 4, 5]
        '''
        super(Cell, self).__init__()
        self.preprocess = torch.nn.ModuleDict()
        if prev_reduce:
            self.preprocess['0'] = FactorizedReduce(in_channels=prev_channels, out_channels=channels, affine=affine)
        else:
            self.preprocess['0'] = ReLUConvBN(in_channels=prev_channels, out_channels=channels, kernel_size=1,
                                              stride=1, padding=0, affine=affine)
        self.preprocess['1'] = ReLUConvBN(in_channels=in_channels, out_channels=channels, kernel_size=1,
                                          stride=1, padding=0, affine=affine)

        self.ops = torch.nn.ModuleDict()
        for path in operations:
            from_ = str(path['from'])
            to_ = str(path['to'])
            op = path['op']
            if to_ not in self.ops:
                self.ops[to_] = torch.nn.ModuleDict()
            if from_ not in self.ops[to_]:
                self.ops[to_][from_] = torch.nn.ModuleDict()
            self.ops[to_][from_] = op
        self.nodes = [str(n) for n in nodes]
        self.drop_path = DropPath(drop_prob=0.0)

    def forward(self, *xs):
        xs = xs[0] if isinstance(xs, tuple) and len(xs) == 1 else xs
        x, x_prev = xs
        x_prev = x_prev if x_prev is not None else x

        inputs = OrderedDict([
            ('0', self.preprocess['0'](x_prev)),
            ('1', self.preprocess['1'](x))
        ])

        for to_, node in self.ops.items():
            out_tensors = torch.stack([
                self.drop_path(op(inputs[from_])) if not isinstance(op, Identity) else op(inputs[from_])
                for from_, op in node.items()
            ])
            inputs[to_] = torch.sum(out_tensors, dim=0)

        x = torch.cat([tensor for idx, tensor in inputs.items() if idx in self.nodes], dim=1)
        return x

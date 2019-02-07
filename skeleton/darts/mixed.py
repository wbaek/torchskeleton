# -*- coding: utf-8 -*-
# pylint: disable=arguments-differ, abstract-method
from __future__ import absolute_import
import logging

from theconf import Config as C
import torch
import torch.nn.functional as F

from .operations import Operations


LOGGER = logging.getLogger(__name__)


class Mixed(torch.nn.Module):
    def __init__(self, names, channels, stride=1, affine=True, alpha=None, tau=1.0):
        super(Mixed, self).__init__()

        self.ops = torch.nn.ModuleList([
            Operations.create(name, channels, stride=stride, affine=affine) for name in names
        ])
        self.alpha = torch.nn.Parameter(torch.Tensor(len(names))) if alpha is None else alpha
        if alpha is None:
            self.reset_parameters()
        self.tau = tau
        self._probs = None

    def reset_parameters(self):
        initial = C.get().conf.get('architecture', {}).get('alphas', {}).get('initial', 'constant')
        if initial == 'random':
            torch.nn.init.normal_(self.alpha, mean=0.0, std=1e-3)
        elif initial == 'random1.0':
            torch.nn.init.normal_(self.alpha, mean=1.0, std=1e-3)
        else:
            torch.nn.init.constant_(self.alpha, 1e-3)

    def update_probs(self):
        self._probs = F.softmax(self.alpha / self.tau, dim=0).view(1, -1)
        return self

    @property
    def probs(self):
        if self._probs is None:
            self.update_probs()
        return self._probs

    def forward(self, x):
        if self.training:
            out_tensors = torch.stack([op(x) for op in self.ops])
            out_shape = out_tensors.shape[1:]
            out = torch.mm(self.probs, out_tensors.view(len(self.ops), -1)).view(out_shape)
        else:
            idx = torch.argmax(self.probs, 1).item()
            out = self.ops[idx](x)
        return out


class DAG:
    @staticmethod
    def create(fn, names, steps, channels, stride=1, affine=True, alpha={}, tau=1.0):  #pylint: disable=dangerous-default-value
        operations = []
        for to_ in range(2, steps+2):
            for from_ in range(to_):
                key = '%s-%s' % (from_, to_)
                stride_ = stride if from_ in [0, 1] else 1
                operations.append({
                    'to': to_,
                    'from': from_,
                    'op': fn(names, channels, stride_, affine,
                             alpha=alpha.get(key, None), tau=tau)
                })

        rv_alpha = {}
        for op in operations:
            key = '%s-%s' % (op['from'], op['to'])
            rv_alpha[key] = op['op'].alpha

        return operations, rv_alpha

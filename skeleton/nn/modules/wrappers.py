# -*- coding: utf-8 -*-
# pylint: disable=arguments-differ
from __future__ import absolute_import
import logging
from collections import OrderedDict

import torch

LOGGER = logging.getLogger(__name__)


class Identity(torch.nn.Module):
    def forward(self, x):
        return x


class FactorizedReduce(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, affine=True):
        assert stride == 2
        assert out_channels % 4 == 0
        super(FactorizedReduce, self).__init__()
        out_channels_half = out_channels // 4

        self.conv = torch.nn.Conv2d(in_channels, out_channels_half, kernel_size=1, stride=stride, padding=0, bias=False)
        self.post = torch.nn.Sequential(
            torch.nn.BatchNorm2d(out_channels, affine=affine),
        )

    def forward(self, x):
        x = torch.cat([
            self.conv(x),
            self.conv(x[:, :, 1:, 0:]),
            self.conv(x[:, :, 0:, 1:]),
            self.conv(x[:, :, 1:, 1:])
        ], dim=1)
        x = self.post(x)
        return x


class Mul(torch.nn.Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight

    def forward(self, x):
        return x * self.weight


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Concat(torch.nn.Module):
    def __init__(self, dim=1):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, *xs):
        return torch.cat(xs, dim=self.dim)


class MergeSum(torch.nn.Module):
    def forward(self, *xs):
        if isinstance(xs[0], (tuple, list)):
            xs = xs[0]
        return torch.sum(torch.stack(xs), dim=0)


class Split(torch.nn.Module):
    def __init__(self, *modules):
        super(Split, self).__init__()
        if len(modules) == 1 and isinstance(modules[0], OrderedDict):
            for key, module in modules[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(modules):
                self.add_module(str(idx), module)

    def forward(self, x):
        return tuple([m(x) for m in self._modules.values()])


class DelayedPass(torch.nn.Module):
    def __init__(self):
        super(DelayedPass, self).__init__()
        self.register_buffer('keep', None)

    def forward(self, x):
        rv = self.keep  # pylint: disable=access-member-before-definition
        self.keep = x
        return rv


class DropPath(torch.nn.Module):
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.training and self.drop_prob > 0.:
            keep_prob = 1. - self.drop_prob
            mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
            x.div_(keep_prob)
            x.mul_(mask)
        return x


class KeepByPass(torch.nn.Module):
    def __init__(self):
        super(KeepByPass, self).__init__()
        self.x = None
        self.info = {}

    def forward(self, x):
        self.x = x
        return x

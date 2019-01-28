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
    def __init__(self, in_channels, out_channels, stride=2):
        assert stride == 2
        assert out_channels % 2 == 0
        super(FactorizedReduce, self).__init__()
        out_channels_half = out_channels // 2

        self.conv = torch.nn.Conv2d(in_channels, out_channels_half, kernel_size=1, stride=stride, padding=0, bias=False)
        self.post = torch.nn.Sequential(
            torch.nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = torch.cat([self.conv(x), self.conv(x[:, :, 1:, 1:])], dim=1)
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
    def __init__(self, num_inputs):
        super(MergeSum, self).__init__()
        self.register_buffer('weight', torch.ones([1, num_inputs], dtype=torch.float32))

    def forward(self, *xs):
        if isinstance(xs[0], (tuple, list)):
            xs = xs[0]

        original_shape = xs[0].shape
        num_inputs = len(xs)
        return torch.mm(self.weight, torch.stack(xs).view(num_inputs, -1)).view(original_shape)


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
        return [m(x) for m in self._modules.values()]

# -*- coding: utf-8 -*-
# pylint: disable=arguments-differ
from __future__ import absolute_import
import logging

import torch
from torch.nn import functional as F

LOGGER = logging.getLogger(__name__)


class Identity(torch.nn.Module):
    def forward(self, x):
        return x


class Mul(torch.nn.Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight

    def forward(self, x):
        return x * self.weight


class Add(torch.nn.Module):
    def forward(self, x, y):
        return x + y


class Concat(torch.nn.Module):
    def __init__(self, dim=1):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, *xs):
        return torch.cat(xs, dim=self.dim)


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class GlobalPool(torch.nn.Module):
    def __init__(self, method=F.adaptive_avg_pool2d):
        super(GlobalPool, self).__init__()
        self.method = method

    def forward(self, x):
        x = self.method(x, (1, 1))
        return x

# -*- coding: utf-8 -*-
# pylint: disable=arguments-differ, abstract-method
from __future__ import absolute_import
import logging

import torch


LOGGER = logging.getLogger(__name__)


class AccuracyMany(torch.nn.Module):
    def __init__(self, topk=(1,)):
        super(AccuracyMany, self).__init__()
        self.topk = topk

    def forward(self, output, target):
        with torch.no_grad():
            maxk = max(self.topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in self.topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(1.0 / batch_size))
        return res


class Accuracy(AccuracyMany):
    def __init__(self, topk=1, scale=1.0):
        super(Accuracy, self).__init__((topk,))
        self.scale = scale

    def forward(self, output, target):
        res = super(Accuracy, self).forward(output, target)
        return res[0] * self.scale

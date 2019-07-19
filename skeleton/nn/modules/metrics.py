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


def to_onehot(labels, shape):
    onehot = torch.zeros(*shape)
    onehot.scatter_(1, labels.unsqueeze(1), 1)
    return onehot


class Fscore(torch.nn.Module):
    def __init__(self, threshold=0.5, beta=1, eps=1e-9):
        super(Fscore, self).__init__()
        self.threshold = threshold
        self.beta = beta
        self.eps = eps

    def forward(self, output, target):
        with torch.no_grad():
            beta2 = self.beta ** 2

            if output.shape != target.shape and target.dtype == torch.long:
                target = to_onehot(target, output.shape).to(device=target.device)

            y_pred = torch.ge(output.float(), self.threshold).float()
            y_true = target.float()

            true_positive = (y_pred * y_true).sum(dim=1)
            precision = true_positive.div(y_pred.sum(dim=1).add(self.eps))
            recall = true_positive.div(y_true.sum(dim=1).add(self.eps))

        return {
            'fscore': torch.mean((precision * recall).div(precision.mul(beta2) + recall + self.eps).mul(1 + beta2)),
            'precision': torch.mean(precision),
            'recall': torch.mean(recall)
        }

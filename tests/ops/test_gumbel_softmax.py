# -*- coding: utf-8 -*-
import torch
from torch.nn import functional as F

from skeleton.ops.gumbel_softmax import gumbel_softmax, sampling_gumbel_softmax


def test_gumbel_softmax():
    logits = torch.Tensor([1.0, 3.0, 7.0, 1.0, 5.0])

    y = gumbel_softmax(logits.view(1, -1), beta=0.).view(-1)
    assert float((y - F.softmax(logits, dim=0)).norm()) < 0.0000001


def test_gumbel_softmax_tau():
    logits = torch.Tensor([1.0, 3.0, 7.0, 1.0, 5.0])

    y = gumbel_softmax(logits.view(1, -1), tau=10., beta=0.).view(-1)
    assert float((y - F.softmax(logits / 10., dim=0)).norm()) < 0.0000001


def test_gumbel_softmax_beta():
    logits = torch.Tensor([1.0, 3.0, 7.0, 1.0, 5.0])

    y = [gumbel_softmax(logits.view(1, -1), beta=0.1).view(-1) for i in range(1000)]
    y = torch.mean(torch.stack(y), dim=0)
    assert 0.0001 < float((y - F.softmax(logits, dim=0)).norm()) < 0.01

    y = [gumbel_softmax(logits.view(1, -1), beta=0.5).view(-1) for i in range(1000)]
    y = torch.mean(torch.stack(y), dim=0)
    assert 0.01 < float((y - F.softmax(logits, dim=0)).norm()) < 0.1

    y = [gumbel_softmax(logits.view(1, -1), beta=1.0).view(-1) for i in range(1000)]
    y = torch.mean(torch.stack(y), dim=0)
    assert 0.05 < float((y - F.softmax(logits, dim=0)).norm()) < 0.2


def test_sampling_gumbel_softmax():
    logits = torch.Tensor([1.0, 3.0, 7.0, 0.0, 5.0])

    y = [sampling_gumbel_softmax(logits.view(1, -1), 1, beta=0., hard=True) for i in range(1000)]
    y = torch.mean(torch.stack(y), dim=0)
    assert float((y - torch.Tensor([0, 0, 1, 0, 0])).norm()) < 0.3

    y = [sampling_gumbel_softmax(logits.view(1, -1), 2, beta=0., hard=False) for i in range(1000)]
    y = torch.mean(torch.stack(y), dim=0)
    assert float((y - F.softmax(logits, dim=0)).norm()) < 0.05


    y = [sampling_gumbel_softmax(logits.view(1, -1), 2, beta=0., hard=True) for i in range(1000)]
    y = torch.mean(torch.stack(y), dim=0)
    assert float((y - torch.Tensor([0, 0, 1, 0, 1])).norm()) < 0.3

    y = [sampling_gumbel_softmax(logits.view(1, -1), 2, beta=0., hard=False) for i in range(1000)]
    y = torch.mean(torch.stack(y), dim=0)
    assert float((y - F.softmax(logits, dim=0)).norm()) < 0.04


    y = [sampling_gumbel_softmax(logits.view(1, -1), 3, beta=0., hard=True) for i in range(1000)]
    y = torch.mean(torch.stack(y), dim=0)
    assert float((y - torch.Tensor([0, 1, 1, 0, 1])).norm()) < 0.3

    y = [sampling_gumbel_softmax(logits.view(1, -1), 3, beta=0., hard=False) for i in range(1000)]
    y = torch.mean(torch.stack(y), dim=0)
    assert float((y - F.softmax(logits, dim=0)).norm()) < 0.01


def test_sampling_gumbel_softmax_with_beta():
    logits = torch.Tensor([1.0, 3.0, 7.0, 0.0, 5.0])

    y = [sampling_gumbel_softmax(logits.view(1, -1), 1, beta=1., hard=True) for i in range(1000)]
    y = torch.mean(torch.stack(y), dim=0)
    assert float((y - torch.Tensor([0, 0, 1, 0, 0])).norm()) < 0.4

    y = [sampling_gumbel_softmax(logits.view(1, -1), 2, beta=1., hard=True) for i in range(1000)]
    y = torch.mean(torch.stack(y), dim=0)
    assert float((y - torch.Tensor([0, 0, 1, 0, 1])).norm()) < 0.4

    y = [sampling_gumbel_softmax(logits.view(1, -1), 3, beta=1., hard=True) for i in range(1000)]
    y = torch.mean(torch.stack(y), dim=0)
    assert float((y - torch.Tensor([0, 1, 1, 0, 1])).norm()) < 0.4

# -*- coding: utf-8 -*-
# pylint: disable=arguments-differ, abstract-method
from __future__ import absolute_import
import logging

import torch
import numpy as np


LOGGER = logging.getLogger(__name__)


def rand_bbox(size, lam):
    W = size[3]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class CutMix(torch.nn.Module):
    def __init__(self, module, criterion, beta):
        super(CutMix, self).__init__()
        self.module = module
        self.criterion = criterion
        self.beta = beta

    def forward(self, input, target):  # pylint: disable=redefined-builtin
        if self.training:
            beta_lambda = np.random.beta(self.beta, self.beta)
            rand_index = torch.randperm(input.size()[0]).to(device=input.device)

            target_a = target
            target_b = target[rand_index]

            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), beta_lambda)
            input[:, :, bby1:bby2, bbx1:bbx2] = input[rand_index, :, bby1:bby2, bbx1:bbx2]

            logits = self.module(input)
            loss = self.criterion(logits, target_a) * beta_lambda + self.criterion(logits, target_b) * (1 - beta_lambda)
        else:
            logits = self.module(input)
            loss = self.criterion(logits, target)
        return logits, loss

# -*- coding: utf-8 -*-
# pylint: disable=arguments-differ, abstract-method
from __future__ import absolute_import
import logging
from collections import OrderedDict

import torch

from ..nn.modules import TraceModule
from ..nn import Split, Identity, Flatten, DelayedPass, KeepByPass
from .cell import Cell


LOGGER = logging.getLogger(__name__)


class DartsBaseNet(TraceModule):
    def __init__(self, channels=32, steps=4, depth=20, num_classes=10):
        super(DartsBaseNet, self).__init__()

        stem_multiplier = 3
        out_channels = channels * stem_multiplier
        layers = [
            ('stem', torch.nn.Sequential(
                torch.nn.Conv2d(3, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(out_channels),
            )),
        ]

        self.delayed_pass = DelayedPass()
        self.auxiliary_keep = KeepByPass()

        reduce_prev, reduce_curr = False, False
        prev_channels, in_channels = out_channels, out_channels
        for i in range(depth):
            reduce_prev, reduce_curr = reduce_curr, i in [depth // 3, 2 * depth // 3]
            prev_channels, in_channels, channels = in_channels, out_channels, (channels * (1 if not reduce_curr else 2))
            out_channels = channels * steps

            sequential = [
                Split(OrderedDict([
                    ('curr', Identity()),
                    ('prev', self.delayed_pass)
                ])),
                self.create_cell(channels, in_channels, prev_channels, reduce_curr, reduce_prev)
            ]
            if i == int(2 * depth // 3):
                sequential.append(self.auxiliary_keep)
                self.auxiliary_keep.info.update({
                    'in_channels': out_channels
                })
            layers.append(
                ('layer%02d'%i, torch.nn.Sequential(*sequential))
            )

        layers.append(
            ('global_pool', torch.nn.AdaptiveAvgPool2d((1, 1)))
        )
        layers.append(
            ('linear', torch.nn.Sequential(
                Flatten(),
                torch.nn.Linear(out_channels, num_classes),
            ))
        )
        self.layers = torch.nn.Sequential(OrderedDict(layers))
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    def create_cell(self, channels, in_channels, prev_channels, curr_reduce, prev_reduce):
        raise NotImplementedError()

    def forward(self, inputs, targets=None):  # pylint: disable=arguments-differ
        self.delayed_pass(None)
        self.auxiliary_keep(None)
        logits = self.layers(inputs)

        if targets is None:
            return logits, None
        if targets.device != logits.device:
            targets = targets.to(device=logits.device)

        loss = self.loss_fn(input=logits, target=targets)
        return logits, loss

    def half(self):
        # super(BasicNet, self).half()
        for module in self.children():
            if not isinstance(module, torch.nn.BatchNorm2d):
                module.half()
        return self

# -*- coding: utf-8 -*-
from __future__ import absolute_import
import logging

from tqdm import tqdm
import numpy as np
import torch


LOGGER = logging.getLogger(__name__)


class SimpleTrainer:
    def __init__(self, module, optimizer, loss_fn, metric_fns=None):
        self.module = module
        self.optimizer = optimizer

        self.loss_fn = loss_fn
        self.metric_fns = metric_fns if metric_fns is not None else {}

        self.global_step = 0
        self.global_epoch = 0

    def forward(self, inputs, targets):
        logits = self.module(inputs)
        targets = targets.to(device=logits.device)
        loss = self.loss_fn(logits, targets)

        metrics = {name: m(logits, targets) for name, m in self.metric_fns.items()} if self.metric_fns is not None else {}
        metrics['loss'] = loss
        return logits, metrics

    def step(self, inputs, targets):
        self.optimizer.zero_grad()

        logits, metrics = self.forward(inputs, targets)
        metrics['loss'].backward()

        self.optimizer.step()
        self.global_step += 1
        return logits, metrics

    def epoch(self, loader, is_training=True, summary_writer=None, desc=''):
        steps = len(loader)
        desc = desc if desc else '[%s] [epoch:%04d]' % ('train' if is_training else 'valid', self.global_epoch)
        generator = tqdm(enumerate(loader), total=steps, desc=desc)

        metric_hist = []
        with torch.set_grad_enabled(is_training):
            _ = self.module.train() if is_training else self.module.eval()
            f = self.step if is_training else self.forward

            for batch_idx, (inputs, targets) in generator:
                logits, metrics = f(inputs, targets)

                metrics = {key: float(value) for key, value in metrics.items()}
                metric_hist.append(metrics)
                generator.set_postfix(metrics)

                if summary_writer:
                    '''
                    for key, value in metrics.items():
                        summary.scalar('metrics/%s' % key, value)
                    '''
                    summary_writer.write(self.global_step)

            self.global_epoch += 1 if is_training else 0

        metric_str = [
            '%s: %.4f' % (metric, np.average([m[metric] for m in metric_hist]))
            for metric in metric_hist[0].keys()
        ]
        LOGGER.info('%s average %s', desc, ', '.join(metric_str))
        return self

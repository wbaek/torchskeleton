# -*- coding: utf-8 -*-
from __future__ import absolute_import
import logging

from tqdm import tqdm
import numpy as np
import torch
from skeleton.summary import summary


LOGGER = logging.getLogger(__name__)


class DartsTrainer:
    def __init__(self, module, optimizers, metric_fns=None):
        '''
        :param module: module.forawrd(inputs, targets) -> logits, loss
        :param optimizers: {'theta': skeleton.optim.ScheduledOptimzer, 'alpha': skeleton.optim.ScheduledOptimzer}
        :param metric_fns: {key: fn}, fn(inputs, targets) -> ScalarTensor
        '''
        self.module = module
        self.optimizers = optimizers

        self.metric_fns = metric_fns if metric_fns is not None else {}

        self.global_step = 0
        self.global_epoch = 0

    def forward(self, inputs, targets):
        logits, loss = self.module(inputs, targets)

        metrics = {name: m(logits, targets) for name, m in self.metric_fns.items()} if self.metric_fns is not None else {}
        metrics['loss'] = loss.mean()
        return logits, metrics

    def step(self, mode, inputs, targets):
        self.module.zero_grad()

        logits, metrics = self.forward(inputs, targets)
        if mode == 'alpha':
            metrics['loss'].backward()
        elif mode == 'theta':
            metrics['loss'].backward(retain_graph=True)  # keep forward graph \wo update probs in Mixed
        else:
            raise NotImplementedError('not support mode at %s' % mode)

        self.optimizers[mode].step()
        return logits, metrics

    def train(self, train_loader, valid_loader, desc='', verbose=False):
        steps = len(train_loader) // 2 if train_loader == valid_loader else len(train_loader)
        desc = desc if desc else '[train] [epoch:%04d]' % (self.global_epoch)

        generator = enumerate(train_loader)
        if verbose:
            generator = tqdm(generator, total=steps, desc=desc)
        if train_loader == valid_loader:
            valid_loader = generator
        else:
            valid_loader = enumerate(valid_loader)

        self.module.train()
        metric_hist = {'alpha': [], 'theta': []}
        with torch.set_grad_enabled(True):
            for _, (inputs, targets) in generator:
                _, (inputs_alpha, targets_alpha) = next(iter(valid_loader))
                _, metrics = self.step('alpha', inputs_alpha, targets_alpha)
                metrics = {key: float(value) for key, value in metrics.items()}
                metric_hist['alpha'].append(metrics)

                # change architecture
                if isinstance(self.module, torch.nn.DataParallel):
                    self.module.module.update_probs()
                else:
                    self.module.update_probs()

                _, metrics = self.step('theta', inputs, targets)
                metrics = {key: float(value) for key, value in metrics.items()}
                metric_hist['theta'].append(metrics)

                self.global_step += 1
                if verbose:
                    generator.set_postfix(metrics)

            self.global_epoch += 1

        metric_avg = {metric: np.average([m[metric] for m in metric_hist['alpha']]) for metric in metric_hist['alpha'][0].keys()}
        for key, value in metric_avg.items():
            summary.scalar('alpha', 'metrics/%s' % key, value)
        metric_str = ['%s: %.4f' % (key, value) for key, value in metric_avg.items()]
        LOGGER.info('%s alpha average %s', desc, ', '.join(metric_str))
        metric_avg_alpha = metric_avg

        metric_avg = {metric: np.average([m[metric] for m in metric_hist['theta']]) for metric in metric_hist['theta'][0].keys()}
        for key, value in metric_avg.items():
            summary.scalar('theta', 'metrics/%s' % key, value)
        metric_str = ['%s: %.4f' % (key, value) for key, value in metric_avg.items()]
        LOGGER.info('%s theta average %s', desc, ', '.join(metric_str))
        metric_avg_theta = metric_avg

        return metric_avg_alpha, metric_avg_theta

    def eval(self, tag, loader, desc='', verbose=False):
        steps = len(loader)
        desc = desc if desc else '[%s] [epoch:%04d]' % (tag, self.global_epoch)

        generator = enumerate(loader)
        if verbose:
            generator = tqdm(generator, total=steps, desc=desc)
        self.module.eval()

        metric_hist = []
        with torch.set_grad_enabled(False):
            for _, (inputs, targets) in generator:
                _, metrics = self.forward(inputs, targets)
                metrics = {key: float(value) for key, value in metrics.items()}
                metric_hist.append(metrics)

        metric_avg = {metric: np.average([m[metric] for m in metric_hist]) for metric in metric_hist[0].keys()}
        for key, value in metric_avg.items():
            summary.scalar(tag, 'metrics/%s' % key, value)
        metric_str = ['%s: %.4f' % (key, value) for key, value in metric_avg.items()]
        LOGGER.info('%s average %s', desc, ', '.join(metric_str))
        return metric_avg

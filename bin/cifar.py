# -*- coding: utf-8 -*-
import os
import sys
import logging

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
import skeleton


class BasicNet(skeleton.nn.modules.IOModule, skeleton.nn.modules.MoveToModule):
    def __init__(self, num_classes=10):
        super(BasicNet, self).__init__()
        self.layers = torch.nn.ModuleDict([
            ('conv1', nn.Sequential(
                torch.nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(64),
            )),
            ('conv2', nn.Sequential(
                torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(128),
            )),
            ('conv3', nn.Sequential(
                torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(256),
            )),
            ('global_pool', skeleton.nn.GlobalPool()),
            ('linear', nn.Sequential(skeleton.nn.Flatten(), nn.Linear(256, num_classes))),
        ])

    def forward(self, x, verbose=False):  # pylint: disable=arguments-differ
        if verbose:
            print('%20s shape after %s' % (list(x.size()), 'inputs'))
        for name, layer in self.layers.items():
            x = layer.forward(x)
            if verbose:
                print('%20s shape after %s' % (list(x.size()), name))
        return x


def main(args):
    logging.info('args: %s', args)
    device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu', 0)

    batch_size = args.batch
    train_loader, valid_loader, data_shape = skeleton.datasets.Cifar.loader(batch_size, args.num_class)

    model = BasicNet(args.num_class).to(device=device)
    if torch.cuda.is_available():
        model.half()
    model(torch.Tensor(*data_shape[0]), verbose=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4 * batch_size, momentum=0.9)

    for epoch in range(args.epoch):
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            logging.info('[train] [epoch:%04d/%04d] [step:%04d/%04d] loss: %.5f', epoch, args.epoch, batch_idx, len(train_loader), float(loss))

        with torch.no_grad():
            losses = [float(F.cross_entropy(model(inputs), targets)) for inputs, targets in valid_loader]
            loss = np.average(losses)
            logging.info('[vaild] [epoch:%04d/%04d]                  loss: %.5f', epoch, args.epoch, loss)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--num-class', type=int, default=10, help='10 or 100')
    parser.add_argument('-b', '--batch', type=int, default=128)
    parser.add_argument('-e', '--epoch', type=int, default=25)

    parser.add_argument('--gpus', type=int, default=torch.cuda.device_count())

    parser.add_argument('--log-filename', type=str, default='')
    parser.add_argument('--debug', action='store_true')
    parsed_args = parser.parse_args()

    log_format = '[%(asctime)s %(levelname)s] %(message)s'
    level = logging.DEBUG if parsed_args.debug else logging.INFO
    if not parsed_args.log_filename:
        logging.basicConfig(level=level, format=log_format, stream=sys.stderr)
    else:
        logging.basicConfig(level=level, format=log_format, filename=parsed_args.log_filename)

    main(parsed_args)

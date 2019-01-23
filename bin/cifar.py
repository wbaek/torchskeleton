# -*- coding: utf-8 -*-
import os
import sys
import logging
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
import skeleton


torch.backends.cudnn.benchmark = True


class BasicNet(skeleton.nn.modules.IOModule):
    def __init__(self, num_classes=10):
        super(BasicNet, self).__init__()
        self.handles = []
        self.layers = torch.nn.Sequential(OrderedDict([
            ('embed', nn.Sequential(
                torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(inplace=True),
            )),
            ('conv1', nn.Sequential(
                torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                skeleton.nn.Split(OrderedDict([
                    ('skip', skeleton.nn.Identity()),
                    ('deep', torch.nn.Sequential(
                        torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                        torch.nn.BatchNorm2d(128),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                        torch.nn.BatchNorm2d(128),
                        torch.nn.ReLU(inplace=True),
                    ))
                ])),
                skeleton.nn.MergeSum()
            )),
            ('conv2', nn.Sequential(
                torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(256),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
            )),
            ('conv3', nn.Sequential(
                torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(512),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                skeleton.nn.Split(OrderedDict([
                    ('skip', skeleton.nn.Identity()),
                    ('deep', torch.nn.Sequential(
                        torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
                        torch.nn.BatchNorm2d(512),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
                        torch.nn.BatchNorm2d(512),
                        torch.nn.ReLU(inplace=True),
                    ))
                ])),
                skeleton.nn.MergeSum()
            )),
            ('global_pool', nn.AdaptiveAvgPool2d((1, 1))),
            ('linear', nn.Sequential(
                skeleton.nn.Flatten(),
                nn.Linear(512, num_classes),
                skeleton.nn.Mul(0.125),
            )),
        ]))

    def forward(self, x, verbose=False):  # pylint: disable=arguments-differ
        return self.layers(x)

    def print_architecture(self):
        for name, module in self.layers.named_modules():
            if name is '':
                continue
            split = name.split('.')
            indent = '\t' * (len(split) - 1)
            class_name = module.__class__.__name__ if module.__class__.__name__ is not 'Sequential' else ''
            print(indent, split[-1], class_name)

    def register_verbose_hooks(self):
        def verbose(module, inputs):
            input_shape = inputs[0].shape if len(inputs[0]) == 1 else [len(inputs[0])] + list(inputs[0][0].shape)
            print('%20s shape input to %s' % (list(input_shape), type(module)))

        def register(m):
            handle = m.register_forward_pre_hook(verbose)
            self.handles.append(handle)

        self.apply(register)
        return self

    def remove_verbose_hooks(self):
        [h.remove() for h in self.handles]
        self.handles = []
        return self


def main(args):
    logging.info('args: %s', args)
    device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu', 0)

    batch_size = args.batch
    train_loader, valid_loader, test_loader, data_shape = skeleton.datasets.Cifar.loader(batch_size, args.num_class, cv_ratio=0.1)

    model = BasicNet(args.num_class).to(device=device)

    print('---------- architecture ---------- ')
    model.print_architecture()

    print('---------- forward steps ---------- ')
    model.register_verbose_hooks()
    model(torch.Tensor(*data_shape[0]).to(device=device), verbose=True)
    model.remove_verbose_hooks()

    model = torch.nn.DataParallel(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4 * batch_size, momentum=0.9)

    trainer = skeleton.trainers.SimpleTrainer(
        model,
        optimizer,
        loss_fn=F.cross_entropy,
        metric_fns={
            'accuracy_top1': skeleton.trainers.metrics.Accuracy(topk=1),
            'accuracy_top5': skeleton.trainers.metrics.Accuracy(topk=5),
        }
    )
    for epoch in range(1, args.epoch):
        trainer.epoch(train_loader, is_training=True)
        trainer.epoch(valid_loader, is_training=False)
        if epoch % 10 == 0:
            trainer.epoch(test_loader, is_training=False, desc='[test] [epoch:%04d]' % epoch)
    trainer.epoch(test_loader, is_training=False, desc='[test]' % epoch)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--num-class', type=int, default=10, help='10 or 100')
    parser.add_argument('-b', '--batch', type=int, default=32)
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

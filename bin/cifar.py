# -*- coding: utf-8 -*-
import os
import sys
import logging
import random
from collections import OrderedDict

import numpy as np
import torch
torch.backends.cudnn.benchmark = True

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
import skeleton


class BasicNet(skeleton.nn.modules.VerboseModule):
    def __init__(self, num_classes=10):
        super(BasicNet, self).__init__()
        self.handles = []
        self.layers = torch.nn.Sequential(OrderedDict([
            ('embed', torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(inplace=True),
            )),
            ('conv1', torch.nn.Sequential(
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
                skeleton.nn.MergeSum(num_inputs=2)
            )),
            ('conv2', torch.nn.Sequential(
                torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(256),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
            )),
            ('conv3', torch.nn.Sequential(
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
                skeleton.nn.MergeSum(num_inputs=2)
            )),
            ('global_pool', torch.nn.AdaptiveAvgPool2d((1, 1))),
            ('linear', torch.nn.Sequential(
                skeleton.nn.Flatten(),
                torch.nn.Linear(512, num_classes),
            )),
        ]))
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    def forward(self, inputs, targets=None):  # pylint: disable=arguments-differ
        logits = self.layers(inputs)

        if targets is None:
            return logits
        if targets.device != logits.device:
            targets = targets.to(device=logits.device)

        loss = self.loss_fn(input=logits, target=targets)
        return logits, loss

    def half(self):
        #super(BasicNet, self).half()
        self.is_half = True
        for module in self.children():
            if not isinstance(module, torch.nn.BatchNorm2d):
                module.half()
        return self


def main(args):
    logging.info('args: %s', args)
    device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu', 0)
    random.seed(0xC0FFEE)
    np.random.seed(0xC0FFEE)
    torch.manual_seed(0xC0FFEE)

    batch_size = args.batch * args.gpus
    train_loader, test_loader, data_shape = skeleton.datasets.Cifar.loader(
        batch_size, args.num_class,
        cv_ratio=0.0, cutout_length=8
    )

    model = BasicNet(args.num_class).to(device=device).half()
    move_to_device_hook_handle = model.register_forward_pre_hook(
        skeleton.nn.hooks.MoveToHook.get_forward_pre_hook(device=device, half=True)
    )
    if args.debug:
        print('---------- architecture ---------- ')
        model.print_architecture()
        print('---------- forward steps ---------- ')
        model.register_verbose_hooks()
        _ = model(torch.Tensor(*data_shape[0]))
        model.remove_verbose_hooks()
        print('---------- done ---------- ')

    optimizer = skeleton.optim.ScheduledOptimzer(
        [p for p in model.parameters() if p.requires_grad],
        torch.optim.SGD,
        steps_per_epoch=len(train_loader),
        lr=lambda e: float(np.interp([e], [0, 5, args.epoch+1], [0, 0.0001, 0])) * batch_size,
        momentum=0.9, weight_decay=1e-5 * batch_size, nesterov=True
    )

    if args.gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.gpus)), output_device=0)
        move_to_device_hook_handle.remove()

    trainer = skeleton.trainers.SimpleTrainer(
        model,
        optimizer,
        metric_fns={
            'accuracy_top1': skeleton.trainers.metrics.Accuracy(topk=1),
            #'accuracy_top5': skeleton.trainers.metrics.Accuracy(topk=5),
        }
    )
    trainer.warmup(
        torch.Tensor(np.random.rand(*data_shape[0])),
        torch.LongTensor(np.random.randint(0, 10, data_shape[1][0]))
    )
    for epoch in range(1, args.epoch):
        trainer.epoch(train_loader, is_training=True, verbose=args.debug)
        if epoch % 10 == 0:
            trainer.epoch(test_loader, is_training=False, verbose=args.debug)
    trainer.epoch(test_loader, is_training=False, verbose=args.debug, desc='[final]')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--num-class', type=int, default=10, help='10 or 100')
    parser.add_argument('-b', '--batch', type=int, default=256)
    parser.add_argument('-e', '--epoch', type=int, default=25)
    parser.add_argument('--gpus', type=int, default=1)#torch.cuda.device_count())

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

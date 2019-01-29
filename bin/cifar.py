# -*- coding: utf-8 -*-
import os
import sys
import logging
import random
import datetime
import shutil
from collections import OrderedDict

import numpy as np
import torch
torch.backends.cudnn.benchmark = True

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
import skeleton


class BasicNet(skeleton.nn.modules.TraceModule):
    def __init__(self, depth=8, num_classes=10):
        super(BasicNet, self).__init__()
        self.handles = []
        layers = [
            ('embed', torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(64),
                torch.nn.LeakyReLU(inplace=True),
            )),
        ]

        stride = 1
        in_channels, out_channels = 64, 64
        for i in range(1, depth+1):
            if i in [1*depth//4, 2*depth//4, 3*depth//4]:
                stride = 2
                out_channels *= 2
            layers.append(
                ('conv%02d'%i, torch.nn.Sequential(
                    skeleton.nn.Split(OrderedDict([
                        ('skip', skeleton.nn.Identity() if in_channels == out_channels else skeleton.nn.FactorizedReduce(in_channels, out_channels)),
                        ('deep', torch.nn.Sequential(
                            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.LeakyReLU(inplace=True),
                            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                            torch.nn.BatchNorm2d(out_channels),
                        ))
                    ])),
                    skeleton.nn.MergeSum(num_inputs=2),
                    torch.nn.LeakyReLU(inplace=True),
                ))
            )
            stride, in_channels, out_channels = 1, out_channels, out_channels

        layers.append(
            ('global_pool', torch.nn.AdaptiveAvgPool2d((1, 1)))
        )
        layers.append(
            ('linear', torch.nn.Sequential(
                skeleton.nn.Flatten(),
                torch.nn.Linear(out_channels, num_classes),
            ))
        )
        self.layers = torch.nn.Sequential(OrderedDict(layers))
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets=None):  # pylint: disable=arguments-differ
        logits = self.layers(inputs)

        if targets is None:
            return logits
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


def main(args):
    random.seed(0xC0FFEE)
    np.random.seed(0xC0FFEE)
    torch.manual_seed(0xC0FFEE)
    logging.info('args: %s', args)
    device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu', 0)

    if args.base_dir is not None:
        writer = skeleton.summary.FileWriter(args.base_dir)

    batch_size = args.batch * args.gpus
    train_loader, test_loader, data_shape = skeleton.datasets.Cifar.loader(
        batch_size, args.num_class,
        cv_ratio=0.0, cutout_length=8
    )

    model = BasicNet(depth=args.depth, num_classes=args.num_class)
    model = torch.nn.DataParallel(model, device_ids=list(range(args.gpus)), output_device=0)

    model.to(device=device)
    model.register_forward_pre_hook(skeleton.nn.hooks.MoveToHook.get_forward_pre_hook(device=device, half=False))

    print('---------- architecture ---------- ')
    handle = model.module.register_forward_pre_hook(skeleton.nn.hooks.MoveToHook.get_forward_pre_hook(device=device, half=False))
    model.module.register_trace_hooks()
    _ = model.module(torch.Tensor(*data_shape[0]))
    model.module.remove_trace_hooks()
    handle.remove()

    model_architecture = model.module.print_trace()
    skeleton.summary.text('train', 'architecture', model_architecture.replace('\n', '<BR/>').replace(' ', '&nbsp;'))
    writer.write(0)
    print('---------- done ---------- ')

    optimizer = skeleton.optim.ScheduledOptimzer(
        [p for p in model.parameters() if p.requires_grad],
        torch.optim.SGD,
        steps_per_epoch=len(train_loader),
        lr=lambda e: float(np.interp([e], [0, 5, args.epoch+1], [0, 0.0001, 0])) * batch_size,
        momentum=0.9, weight_decay=1e-6 * batch_size, nesterov=True
    )

    trainer = skeleton.trainers.SimpleTrainer(
        model,
        optimizer,
        metric_fns={
            'accuracy_top1': skeleton.trainers.metrics.Accuracy(topk=1),
            'accuracy_top5': skeleton.trainers.metrics.Accuracy(topk=5),
        }
    )
    trainer.warmup(
        torch.Tensor(np.random.rand(*data_shape[0])),
        torch.LongTensor(np.random.randint(0, 10, data_shape[1][0]))
    )
    for epoch in range(1, args.epoch):
        trainer.epoch('train', train_loader, is_training=True, verbose=args.debug)
        trainer.epoch('valid', test_loader, is_training=False, verbose=args.debug)
        writer.write(epoch)
    trainer.epoch('valid', test_loader, is_training=False, verbose=args.debug, desc='[final]')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--depth', type=int, default=8)
    parser.add_argument('-c', '--num-class', type=int, default=10, help='10 or 100')
    parser.add_argument('-b', '--batch', type=int, default=256)
    parser.add_argument('-e', '--epoch', type=int, default=30)
    parser.add_argument('--gpus', type=int, default=torch.cuda.device_count())

    parser.add_argument('--base-dir', type=str, required=None)
    parser.add_argument('--name', type=str, required=None)

    parser.add_argument('--log-filename', type=str, default='')
    parser.add_argument('--debug', action='store_true')
    parsed_args = parser.parse_args()

    log_format = '[%(asctime)s %(levelname)s] %(message)s'
    level = logging.DEBUG if parsed_args.debug else logging.INFO
    if not parsed_args.log_filename:
        logging.basicConfig(level=level, format=log_format, stream=sys.stderr)
    else:
        logging.basicConfig(level=level, format=log_format, filename=parsed_args.log_filename)

    name = 'modified_resnet18'
    name += ('_' + parsed_args.name) if parsed_args.name is not None else ''
    if parsed_args.base_dir is None:
        parsed_args.base_dir = '/'.join([
            '.',
            'experiments',
            'cifar' + str(parsed_args.num_class),
            'reference',
            name,
            datetime.datetime.now().strftime('%Y%m%d'),
            datetime.datetime.now().strftime('%H%M')
        ])

    if os.path.exists(parsed_args.base_dir):
        logging.warning('remove exists folder at %s', parsed_args.base_dir)
        shutil.rmtree(parsed_args.base_dir)
    os.makedirs(parsed_args.base_dir + '/models', exist_ok=True)

    main(parsed_args)

# -*- coding: utf-8 -*-
import os
import sys
import copy
import random
import logging
from collections import OrderedDict

import numpy as np
import torch
torch.backends.cudnn.benchmark = True

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
import skeleton


GENOTYPES = OrderedDict([
    ('normal', [
        {'to': 2, 'from': 0, 'name': 'conv_sep_3'},
        {'to': 2, 'from': 1, 'name': 'conv_sep_3'},
        {'to': 3, 'from': 0, 'name': 'conv_sep_3'},
        {'to': 3, 'from': 1, 'name': 'conv_sep_3'},
        {'to': 4, 'from': 0, 'name': 'skip'},
        {'to': 4, 'from': 1, 'name': 'conv_sep_3'},
        {'to': 5, 'from': 0, 'name': 'skip'},
        {'to': 5, 'from': 1, 'name': 'conv_dil_2_3'},
    ]),
    ('reduce', [
        {'to': 2, 'from': 0, 'name': 'pool_max_3'},
        {'to': 2, 'from': 1, 'name': 'pool_max_3'},
        {'to': 3, 'from': 1, 'name': 'pool_max_3'},
        {'to': 3, 'from': 2, 'name': 'skip'},
        {'to': 4, 'from': 0, 'name': 'pool_max_3'},
        {'to': 4, 'from': 2, 'name': 'skip'},
        {'to': 5, 'from': 1, 'name': 'pool_max_3'},
        {'to': 5, 'from': 2, 'name': 'skip'},
    ])
])


class DartsSearchedNet(skeleton.nn.modules.TraceModule):
    def __init__(self, channels=32, steps=4, depth=20, num_classes=10):
        super(DartsSearchedNet, self).__init__()

        stem_multiplier = 3
        out_channels = channels * stem_multiplier
        layers = [
            ('stem', torch.nn.Sequential(
                torch.nn.Conv2d(3, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(out_channels),
            )),
        ]

        prev_reduce, reduce = False, False
        prev_channels, in_channels, out_channels, channels = out_channels, out_channels, out_channels, channels
        self.delayed_pass = skeleton.nn.DelayedPass(length=1)
        for i in range(depth):
            prev_reduce, reduce = reduce, i in [depth // 3, 2 * depth // 3]
            prev_channels, in_channels, channels = in_channels, out_channels, (channels * (1 if not reduce else 2))
            out_channels = channels * steps
            operations = []
            for path in GENOTYPES['normal' if not reduce else 'reduce']:
                new_path = copy.deepcopy(path)
                stride = 2 if path['from'] in [0, 1] and reduce else 1
                new_path['op'] = skeleton.darts.Operations.create(path['name'], channels, stride=stride, affine=True)
                operations.append(new_path)

            layers.append(
                ('layer%02d'%i, torch.nn.Sequential(
                    skeleton.nn.Split(OrderedDict([
                        ('curr', skeleton.nn.Identity()),
                        ('prev', self.delayed_pass)
                    ])),
                    skeleton.darts.layers.Cell(operations, channels, in_channels, prev_channels,
                                               prev_reduce=prev_reduce, affine=True),
                ))
            )
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
        self.delayed_pass.forward(None)
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
    logging.info('args: %s', args)
    device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu', 0)
    random.seed(0xC0FFEE)
    np.random.seed(0xC0FFEE)
    torch.manual_seed(0xC0FFEE)

    batch_size = args.batch * args.gpus
    train_loader, test_loader, data_shape = skeleton.datasets.Cifar.loader(
        batch_size, args.num_class,
        cv_ratio=0.0, cutout_length=16
    )

    model = DartsSearchedNet(channels=args.init_channels, steps=4, depth=args.depth, num_classes=args.num_class)
    model = torch.nn.DataParallel(model, device_ids=list(range(args.gpus)), output_device=0)

    model.to(device=device)
    model.register_forward_pre_hook(skeleton.nn.hooks.MoveToHook.get_forward_pre_hook(device=device, half=False))
    if args.debug:
        print('---------- architecture ---------- ')
        handle = model.module.register_forward_pre_hook(skeleton.nn.hooks.MoveToHook.get_forward_pre_hook(device=device, half=False))
        model.module.register_trace_hooks()
        _ = model.module(torch.Tensor(np.random.rand(*data_shape[0])))
        model.module.remove_trace_hooks()
        model.module.print_trace()
        handle.remove()
        print('---------- done ---------- ')

    optimizer = skeleton.optim.ScheduledOptimzer(
        [p for p in model.parameters() if p.requires_grad],
        torch.optim.SGD,
        steps_per_epoch=len(train_loader),
        lr=lambda e: float(np.interp([e], [0, args.epoch+1], [0.025, 0])),
        momentum=0.9, weight_decay=3e-4, nesterov=True
    )

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
    for _ in range(1, args.epoch):
        trainer.epoch(train_loader, is_training=True, verbose=args.debug)
        if args.debug:
            trainer.epoch(test_loader, is_training=False, verbose=args.debug)
    trainer.epoch(test_loader, is_training=False, verbose=args.debug, desc='[final]')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', type=int, default=20)
    parser.add_argument('--init-channels', type=int, default=32)

    parser.add_argument('-c', '--num-class', type=int, default=10, help='10 or 100')
    parser.add_argument('-b', '--batch', type=int, default=96)
    parser.add_argument('-e', '--epoch', type=int, default=600)
    parser.add_argument('--gpus', type=int, default=torch.cuda.device_count())

    parser.add_argument('--log-filename', type=str, default='')
    parser.add_argument('--debug', action='store_true')
    parsed_args = parser.parse_args()

    parsed_args.depth = 20
    parsed_args.gpus = 1
    parsed_args.debug = True

    log_format = '[%(asctime)s %(levelname)s] %(message)s'
    level = logging.DEBUG if parsed_args.debug else logging.INFO
    if not parsed_args.log_filename:
        logging.basicConfig(level=level, format=log_format, stream=sys.stderr)
    else:
        logging.basicConfig(level=level, format=log_format, filename=parsed_args.log_filename)

    main(parsed_args)

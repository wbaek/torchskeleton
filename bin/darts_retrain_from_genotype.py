# -*- coding: utf-8 -*-
import os
import sys
import copy
import math
import random
import shutil
import datetime
import logging
from collections import OrderedDict

import numpy as np
import torch
torch.backends.cudnn.benchmark = True

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
import skeleton


GENOTYPES = OrderedDict([
    ('normal', {
        'path': [
            {'to': 2, 'from': 0, 'name': 'conv_sep_3'},
            {'to': 2, 'from': 1, 'name': 'conv_sep_3'},
            {'to': 3, 'from': 0, 'name': 'conv_sep_3'},
            {'to': 3, 'from': 1, 'name': 'conv_sep_3'},
            {'to': 4, 'from': 0, 'name': 'skip'},
            {'to': 4, 'from': 1, 'name': 'conv_sep_3'},
            {'to': 5, 'from': 0, 'name': 'skip'},
            {'to': 5, 'from': 1, 'name': 'conv_dil_2_3'},
        ],
        'node': [2, 3, 4, 5]
    }),
    ('reduce', {
        'path': [
            {'to': 2, 'from': 0, 'name': 'pool_max_3'},
            {'to': 2, 'from': 1, 'name': 'pool_max_3'},
            {'to': 3, 'from': 1, 'name': 'pool_max_3'},
            {'to': 3, 'from': 2, 'name': 'skip'},
            {'to': 4, 'from': 0, 'name': 'pool_max_3'},
            {'to': 4, 'from': 2, 'name': 'skip'},
            {'to': 5, 'from': 1, 'name': 'pool_max_3'},
            {'to': 5, 'from': 2, 'name': 'skip'},
        ],
        'node': [2, 3, 4, 5]
    })
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

        reduce_prev, reduce_curr = False, False
        prev_channels, in_channels, out_channels, channels = out_channels, out_channels, out_channels, channels
        self.delayed_pass = skeleton.nn.DelayedPass(length=1)
        self.auxiliary_keep = skeleton.nn.KeepByPass()
        for i in range(depth):
            reduce_prev, reduce_curr = reduce_curr, i in [depth // 3, 2 * depth // 3]
            prev_channels, in_channels, channels = in_channels, out_channels, (channels * (1 if not reduce_curr else 2))
            out_channels = channels * steps
            operations = []
            for path in GENOTYPES['normal' if not reduce_curr else 'reduce']['path']:
                new_path = copy.deepcopy(path)
                stride = 2 if path['from'] in [0, 1] and reduce_curr else 1
                new_path['op'] = skeleton.darts.Operations.create(path['name'], channels, stride=stride, affine=True)
                operations.append(new_path)
            nodes = GENOTYPES['normal' if not reduce_curr else 'reduce']['node']

            sequential = [
                skeleton.nn.Split(OrderedDict([
                    ('curr', skeleton.nn.Identity()),
                    ('prev', self.delayed_pass)
                ])),
                skeleton.darts.layers.Cell(operations, nodes, channels, in_channels, prev_channels,
                                           prev_reduce=reduce_prev, affine=True),
            ]
            if i == int(2 * depth // 3):
                sequential.append(self.auxiliary_keep)
                auxiliary_info = {'idx': i, 'channels': channels, 'out_channels': out_channels}
            layers.append(
                ('layer%02d'%i, torch.nn.Sequential(*sequential))
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
        self.auxiliary = torch.nn.Sequential(
            torch.nn.ReLU(inplace=False),
            torch.nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), # image size = 2 x 2
            torch.nn.Conv2d(auxiliary_info['out_channels'], 128, 1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 768, 2, bias=False),
            torch.nn.BatchNorm2d(768),
            torch.nn.ReLU(inplace=True),
            skeleton.nn.Flatten(),
            torch.nn.Linear(768, num_classes)
        )

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets=None):  # pylint: disable=arguments-differ
        self.delayed_pass.forward(None)
        self.auxiliary_keep(None)
        logits = self.layers(inputs)
        if self.training:
            auxiliary_logits = self.auxiliary(self.auxiliary_keep.x)

        if targets is None:
            return logits
        if targets.device != logits.device:
            targets = targets.to(device=logits.device)

        loss = self.loss_fn(input=logits, target=targets)
        if self.training:
            auxiliary_loss = self.loss_fn(input=auxiliary_logits, target=targets)
            total_loss = loss + (auxiliary_loss * 0.4)
        else:
            total_loss = loss
        return logits, total_loss

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
        cv_ratio=0.0, cutout_length=16
    )

    model = DartsSearchedNet(channels=args.init_channels, steps=4, depth=args.depth, num_classes=args.num_class).train()
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
    print('---------- done. ---------- ')

    def get_lr_cosine_schedule(init_lr, maximum_epoch, eta_min=0):
        def schedule(e):
            return (1 + math.cos(math.pi * e / maximum_epoch)) / \
                   (1 + math.cos(math.pi * (e - 1) / maximum_epoch)) * \
                   (init_lr - eta_min) + eta_min
        return schedule
    optimizer = skeleton.optim.ScheduledOptimzer(
        [p for p in model.parameters() if p.requires_grad],
        torch.optim.SGD,
        steps_per_epoch=len(train_loader),
        lr=get_lr_cosine_schedule(init_lr=0.025, maximum_epoch=args.epoch),
        momentum=0.9, weight_decay=3e-4, nesterov=False
    )

    trainer = skeleton.trainers.SimpleTrainer(
        model,
        optimizer,
        metric_fns={
            'accuracy': skeleton.trainers.metrics.Accuracy(topk=1),
            #'accuracy_top5': skeleton.trainers.metrics.Accuracy(topk=5),
        }
    )
    trainer.warmup(
        torch.Tensor(np.random.rand(*data_shape[0])),
        torch.LongTensor(np.random.randint(0, 10, data_shape[1][0]))
    )
    print('---------- warmup done. ---------- ')

    for epoch in range(args.epoch):
        def apply_drop_prob(module):
            if isinstance(module, skeleton.nn.DropPath):
                drop_prob = 0.2 * epoch / args.epoch  # pylint: disable=cell-var-from-loop
                module.drop_prob = drop_prob
                skeleton.summary.scalar('train', 'annealing/path_drop/drop_prob', drop_prob)
        model.apply(apply_drop_prob)

        metrics_train = trainer.epoch('train', train_loader, is_training=True, verbose=args.debug)
        metrics_valid = trainer.epoch('valid', test_loader, is_training=False, verbose=args.debug)
        writer.write(epoch)
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'metrics': {
                'train': metrics_train,
                'valid': metrics_valid
            }
        }, args.base_dir + '/models/epoch_%04d.pth' % epoch)

    trainer.epoch('valid', test_loader, is_training=False, verbose=args.debug, desc='[final]')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', type=int, default=20)
    parser.add_argument('--init-channels', type=int, default=32)

    parser.add_argument('-c', '--num-class', type=int, default=10, help='10 or 100')
    parser.add_argument('-b', '--batch', type=int, default=96)
    parser.add_argument('-e', '--epoch', type=int, default=600)
    parser.add_argument('--gpus', type=int, default=torch.cuda.device_count())

    parser.add_argument('--base-dir', type=str, required=None)
    parser.add_argument('--name', type=str, required=None)

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

    name = 'genotypes_from_paper'
    name += ('_' + parsed_args.name) if parsed_args.name is not None else ''
    if parsed_args.base_dir is None:
        parsed_args.base_dir = '/'.join([
            '.',
            'experiments',
            'cifar' + str(parsed_args.num_class),
            'darts',
            'retrain',
            name,
            datetime.datetime.now().strftime('%Y%m%d'),
            datetime.datetime.now().strftime('%H%M')
        ])

    if os.path.exists(parsed_args.base_dir):
        logging.warning('remove exists folder at %s', parsed_args.base_dir)
        shutil.rmtree(parsed_args.base_dir)
    os.makedirs(parsed_args.base_dir + '/models', exist_ok=True)

    main(parsed_args)

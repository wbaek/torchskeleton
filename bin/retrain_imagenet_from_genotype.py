# -*- coding: utf-8 -*-
from __future__ import absolute_import
import os
import sys
import copy
import shutil
import datetime
import random
import logging
from collections import OrderedDict

import numpy as np
import torch
torch.backends.cudnn.benchmark = True

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
import skeleton
import genotypes


GENOTYPES = genotypes.ORIGINAL_DARTS


class DartsSearchedImageNet(skeleton.nn.TraceModule, skeleton.nn.ProfileModule):
    def __init__(self, channels=48, steps=4, depth=15, num_classes=1000):
        super(DartsSearchedImageNet, self).__init__()
        self.delayed_pass = skeleton.nn.DelayedPass()
        self.auxiliary_keep = skeleton.nn.KeepByPass()

        stem_multiplier = 3
        out_channels = channels * stem_multiplier
        self.stem0 = torch.nn.Sequential(
            torch.nn.Conv2d(3, out_channels // 2, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels // 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
        )
        self.stem1 = torch.nn.Sequential(
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
        )

        layers = []
        reduce_prev, reduce_curr = False, True
        prev_channels, in_channels = out_channels, out_channels
        for i in range(depth):
            reduce_prev, reduce_curr = reduce_curr, i in [depth // 3, 2 * depth // 3]
            prev_channels, in_channels, channels = in_channels, out_channels, (channels * (1 if not reduce_curr else 2))
            out_channels = channels * steps

            sequential = []
            if i > 0:
                sequential.append(
                    skeleton.nn.Split(OrderedDict([
                        ('curr', skeleton.nn.Identity()),
                        ('prev', self.delayed_pass)
                    ])),
                )
            sequential.append(self.create_cell(channels, in_channels, prev_channels, reduce_curr, reduce_prev))
            if i == int(2 * depth // 3):
                sequential.append(self.auxiliary_keep)
                self.auxiliary_keep.info.update({
                    'in_channels': out_channels
                })
            layers.append(
                ('layer%02d' % i, torch.nn.Sequential(*sequential))
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
            torch.nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False), # image size = 2 x 2
            torch.nn.Conv2d(self.auxiliary_keep.info['in_channels'], 128, 1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 768, 2, bias=False),
            #torch.nn.BatchNorm2d(768),
            torch.nn.ReLU(inplace=True),
            skeleton.nn.Flatten(),
            torch.nn.Linear(768, num_classes)
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.loss_fn_smooth = skeleton.nn.CrossEntropyLabelSmooth(num_classes, 0.1)

    def create_cell(self, channels, in_channels, prev_channels, curr_reduce, prev_reduce):
        operations = []
        for path in GENOTYPES['normal' if not curr_reduce else 'reduce']['path']:
            new_path = copy.deepcopy(path)
            stride = 2 if path['from'] in [0, 1] and curr_reduce else 1
            new_path['op'] = skeleton.darts.Operations.create(path['name'], channels, stride=stride, affine=True)
            operations.append(new_path)
        nodes = GENOTYPES['normal' if not curr_reduce else 'reduce']['node']
        return skeleton.darts.cell.Cell(operations, nodes, channels, in_channels, prev_channels, prev_reduce, affine=True)

    def forward(self, inputs, targets=None):  # pylint: disable=arguments-differ
        self.delayed_pass(None)
        self.auxiliary_keep(None)
        x0 = self.stem0(inputs)
        x1 = self.stem1(x0)
        self.delayed_pass(x1)
        logits = self.layers((x1, x0))

        if targets is None:
            return logits, None
        if self.training:
            loss = self.loss_fn_smooth(input=logits, target=targets)
            auxiliary_logits = self.auxiliary(self.auxiliary_keep.x)
            auxiliary_loss = self.loss_fn_smooth(input=auxiliary_logits, target=targets)
            loss = loss + (auxiliary_loss * 0.4)
        else:
            loss = self.loss_fn(input=logits, target=targets)
        return logits, loss

    def half(self):
        super(DartsSearchedImageNet, self).half()
        for module in self.children():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.float()
        return self


def main(args):
    random.seed(0xC0FFEE)
    np.random.seed(0xC0FFEE)
    torch.manual_seed(0xC0FFEE)
    torch.cuda.manual_seed(0xC0FFEE)
    logging.info('args: %s', args)
    device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu', 0)

    if args.base_dir is not None:
        writer = skeleton.summary.FileWriter(args.base_dir)

    batch_size = args.batch * args.gpus
    train_loader, test_loader, data_shape = skeleton.datasets.Imagenet.loader(batch_size, cv_ratio=0.0, num_workers=48)

    model = DartsSearchedImageNet(channels=args.init_channels, steps=4, depth=args.depth, num_classes=args.num_class)
    model.to(device).train()
    print('---------- architecture ---------- ')
    model.register_trace_hooks()
    _ = model(
        inputs=torch.Tensor(*((2,)+data_shape[0][1:])).to(device),
        targets=torch.LongTensor(np.random.randint(0, 10, (2,))).to(device)
    )
    model.remove_trace_hooks()

    model_architecture = model.print_trace()
    skeleton.summary.text('train', 'architecture', model_architecture.replace('\n', '<BR/>').replace(' ', '&nbsp;'))
    writer.write(0)
    print('---------- profile ---------- ')
    model.eval()
    handle = model.register_forward_pre_hook(
        skeleton.nn.hooks.MoveToHook.get_forward_pre_hook(device=device, half=False))
    model.register_profile_hooks(
        module_filter=lambda name: not any(n in name for n in ['skeleton', 'loss', 'BatchNorm', 'ReLU'])
    )
    _ = model(
        inputs=torch.Tensor(*((1,) + data_shape[0][1:])).to(device)
    )
    model.remove_profile_hooks()
    handle.remove()

    total_params = model.count_parameters(name_filter=lambda name: 'auxiliary' not in name)
    total_flops = model.count_flops(name_filter=lambda name: 'auxiliary' not in name)
    print('#params: %.3f MB' % (total_params / 1e6))
    print('#Flops: %.3f MB' % (total_flops / 1e6))
    print('---------- done. ---------- ')

    scheduler = skeleton.optim.gradual_warm_up(
        skeleton.optim.get_discrete_epoch(
            skeleton.optim.get_step_scheduler(0.1, 1, 0.97)
        ),
        maximum_epoch=5, multiplier=batch_size / 128.0
    )

    optimizer = skeleton.optim.ScheduledOptimzer(
        [p for p in model.parameters() if p.requires_grad],
        torch.optim.SGD,
        steps_per_epoch=len(train_loader),
        clip_grad_max_norm=5.0,
        lr=scheduler,
        momentum=0.9,
        weight_decay=3e-5,
        nesterov=False
    )

    # model.half()
    if args.gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.gpus)), output_device=0)
    model.register_forward_pre_hook(skeleton.nn.hooks.MoveToHook.get_forward_pre_hook(device=device, half=False))
    model.to(device).train()
    trainer = skeleton.trainers.SimpleTrainer(
        model,
        optimizer,
        metric_fns={
            'accuracy': skeleton.trainers.metrics.Accuracy(topk=1, scale=100.0),
            'accuracy_top5': skeleton.trainers.metrics.Accuracy(topk=5, scale=100.0),
        }
    )

    def initialize(module):
        if isinstance(module, torch.nn.BatchNorm2d):
            module.reset_running_stats()
    model.apply(initialize)

    for epoch in range(args.epoch):
        def apply_drop_prob(module):
            if isinstance(module, skeleton.nn.DropPath):
                drop_prob = 0.0 * epoch / args.epoch  # pylint: disable=cell-var-from-loop
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

    metrics = trainer.epoch('valid', test_loader, is_training=False, verbose=args.debug, desc='[final]')
    print(metrics)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', type=int, default=14)
    parser.add_argument('--init-channels', type=int, default=48)

    parser.add_argument('-c', '--num-class', type=int, default=1000)
    parser.add_argument('-b', '--batch', type=int, default=128)
    parser.add_argument('-e', '--epoch', type=int, default=250)
    parser.add_argument('--gpus', type=int, default=torch.cuda.device_count())

    parser.add_argument('--base-dir', type=str, default=None)
    parser.add_argument('--name', type=str, default=None)

    parser.add_argument('--log-filename', type=str, default='')
    parser.add_argument('--debug', action='store_true')
    parsed_args = parser.parse_args()

    parsed_args.debug = True
    parsed_args.batch = 256

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
            'imagenet' + str(parsed_args.num_class),
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

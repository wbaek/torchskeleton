# -*- coding: utf-8 -*-
from __future__ import absolute_import
import os
import sys
import copy
import shutil
import datetime
import random
import logging

import numpy as np
import torch
torch.backends.cudnn.benchmark = True

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
import skeleton
import genotypes


GENOTYPES = genotypes.ORIGINAL_DARTS
# GENOTYPES = genotypes.MAML_FROM_CIFAR10
# GENOTYPES = genotypes.MAML_FROM_CIFAR100
# GENOTYPES = genotypes.MAML_STOPGRAD_FROM_CIFAR10
# GENOTYPES = genotypes.MAML_STOPGRAD_FROM_CIFAR100
# GENOTYPES = genotypes.MAML_FROM_CIFAR10_ALL_INPUT
# GENOTYPES = genotypes.MAML_FROM_CIFAR100_ALL_INPUT
# GENOTYPES = genotypes.MAML_STOPGRAD_FROM_CIFAR10_ALL_INPUT
# GENOTYPES = genotypes.MAML_STOPGRAD_FROM_CIFAR100_ALL_INPUT
# GENOTYPES = genotypes.MAML_CUTOUT_FROM_CIFAR100
# GENOTYPES = genotypes.MAML_REPTILE_FROM_CIFAR100

class DartsSearchedNet(skeleton.darts.models.DartsBaseNet):
    def __init__(self, channels=32, steps=4, depth=20, num_classes=10):
        super(DartsSearchedNet, self).__init__(channels=channels, steps=steps, depth=depth, num_classes=num_classes)
        self.auxiliary = torch.nn.Sequential(
            torch.nn.ReLU(inplace=False),
            torch.nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), # image size = 2 x 2
            torch.nn.Conv2d(self.auxiliary_keep.info['in_channels'], 128, 1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 768, 2, bias=False),
            torch.nn.BatchNorm2d(768),
            torch.nn.ReLU(inplace=True),
            skeleton.nn.Flatten(),
            torch.nn.Linear(768, num_classes)
        )

    def create_cell(self, channels, in_channels, prev_channels, curr_reduce, prev_reduce):
        operations = []
        for path in GENOTYPES['normal' if not curr_reduce else 'reduce']['path']:
            new_path = copy.deepcopy(path)
            stride = 2 if path['from'] in [0, 1] and curr_reduce else 1
            new_path['op'] = skeleton.darts.operations.Operations.create(path['name'], channels, stride=stride, affine=True)
            operations.append(new_path)
        nodes = GENOTYPES['normal' if not curr_reduce else 'reduce']['node']
        return skeleton.darts.cell.Cell(operations, nodes, channels, in_channels, prev_channels, prev_reduce, affine=True)

    def forward(self, inputs, targets=None):  # pylint: disable=arguments-differ
        logits, loss = super(DartsSearchedNet, self).forward(inputs, targets)
        if self.training:
            if loss is not None:
                auxiliary_logits = self.auxiliary(self.auxiliary_keep.x)
                auxiliary_loss = self.loss_fn(input=auxiliary_logits, target=targets)
                loss = loss + (auxiliary_loss * 0.4)
        return logits, loss


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
    train_loader, test_loader, data_shape = skeleton.datasets.Cifar.loader(
        batch_size, args.num_class,
        cv_ratio=0.0, cutout_length=16
    )

    model = DartsSearchedNet(channels=args.init_channels, steps=4, depth=args.depth, num_classes=args.num_class)
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
            skeleton.optim.get_cosine_scheduler(init_lr=0.025, maximum_epoch=args.epoch)
        ),
        maximum_epoch=10, multiplier=batch_size / 96
    )

    optimizer = skeleton.optim.ScheduledOptimzer(
        [p for p in model.parameters() if p.requires_grad],
        torch.optim.SGD,
        steps_per_epoch=len(train_loader),
        clip_grad_max_norm=5.0,
        lr=scheduler,
        momentum=0.9,
        weight_decay=3e-4,
        nesterov=False
    )

    if args.gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.gpus)), output_device=0)
    model.register_forward_pre_hook(skeleton.nn.hooks.MoveToHook.get_forward_pre_hook(device=device, half=False))
    model.to(device).train()
    trainer = skeleton.trainers.SimpleTrainer(
        model,
        optimizer,
        metric_fns={
            'accuracy': skeleton.trainers.metrics.Accuracy(topk=1, scale=100.0),
            #'accuracy_top5': skeleton.trainers.metrics.Accuracy(topk=5),
        }
    )
    trainer.warmup(
        torch.Tensor(np.random.rand(*data_shape[0])),
        torch.LongTensor(np.random.randint(0, 10, data_shape[1][0]))
    )
    print('---------- warmup done. ---------- ')

    def initialize(module):
        if isinstance(module, torch.nn.BatchNorm2d):
            module.reset_running_stats()
    model.apply(initialize)

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

    metrics = trainer.epoch('valid', test_loader, is_training=False, verbose=args.debug, desc='[final]')
    print(metrics)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', type=int, default=20)
    parser.add_argument('--init-channels', type=int, default=36)

    parser.add_argument('-c', '--num-class', type=int, default=10, help='10 or 100')
    parser.add_argument('-b', '--batch', type=int, default=96)
    parser.add_argument('-e', '--epoch', type=int, default=600)
    parser.add_argument('--gpus', type=int, default=torch.cuda.device_count())

    parser.add_argument('--base-dir', type=str, default=None)
    parser.add_argument('--name', type=str, default=None)

    parser.add_argument('--log-filename', type=str, default='')
    parser.add_argument('--debug', action='store_true')
    parsed_args = parser.parse_args()

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

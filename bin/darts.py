# -*- coding: utf-8 -*-
import os
import sys
import random
import shutil
import datetime
import logging

from theconf import Config as C
from theconf import ConfigArgumentParser

import numpy as np
import torch
torch.backends.cudnn.benchmark = True

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
import skeleton


class DartsNASNet(skeleton.darts.models.DartsBaseNet):
    def __init__(self, channels=16, steps=4, depth=8, num_classes=10):
        self.alphas = {'normal': {}, 'reduce': {}}
        self.operations = [
            'zero', 'skip', 'pool_avg_3', 'pool_max_3',
            'conv_sep_3', 'conv_dil_2_3', 'conv_sep_5', 'conv_dil_2_5'
        ]
        super(DartsNASNet, self).__init__(channels=channels, steps=steps, depth=depth, num_classes=num_classes)

    def create_cell(self, channels, in_channels, prev_channels, curr_reduce, prev_reduce):
        stride = 2 if curr_reduce else 1
        cell_type = 'reduce' if curr_reduce else 'normal'
        operations, alphas = skeleton.darts.mixed.DAG.create(
            skeleton.darts.mixed.Mixed,
            self.operations,
            steps=4,
            channels=channels,
            stride=stride,
            affine=False,
            alpha=self.alphas[cell_type]
        )
        self.alphas[cell_type] = alphas
        nodes = [2, 3, 4, 5]
        return skeleton.darts.cell.Cell(operations, nodes, channels, in_channels, prev_channels, prev_reduce, affine=False)


def main(args):
    random.seed(0xC0FFEE)
    np.random.seed(0xC0FFEE)
    torch.manual_seed(0xC0FFEE)
    torch.cuda.manual_seed(0xC0FFEE)
    logging.info('args: %s', args)
    device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu', 0)

    writer = skeleton.summary.FileWriter(args.base_dir, tags=['train', 'theta', 'alpha', 'valid', 'test'])

    batch_size = args.batch * args.gpus
    train_loader, valid_loader, test_loader, data_shape = skeleton.datasets.Cifar.loader(
        batch_size, args.num_class,
        cv_ratio=0.5, cutout_length=0
    )

    # TODO: create NASNetwork from config
    model = DartsNASNet(channels=16, steps=4, depth=8, num_classes=args.num_class)
    model.to(device).train()
    print('---------- architecture ---------- ')
    model.register_trace_hooks()
    _ = model(
        inputs=torch.Tensor(*((2,) + data_shape[0][1:])).to(device),
        targets=torch.LongTensor(np.random.randint(0, 10, (2,))).to(device)
    )
    model.remove_trace_hooks()

    model_architecture = model.print_trace()
    skeleton.summary.text('train', 'models/architecture', model_architecture.replace('\n', '<BR/>').replace(' ', '&nbsp;'))
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
    skeleton.summary.text('train', 'profile/params', '%.3f MB' % (total_params / 1e6))
    skeleton.summary.text('train', 'profile/flops', '%.3f MB' % (total_flops / 1e6))
    print('---------- done. ---------- ')
    writer.write(0)

    scheduler_theta = skeleton.optim.gradual_warm_up(
        skeleton.optim.get_discrete_epoch(
            skeleton.optim.get_cosine_scheduler(init_lr=0.025, maximum_epoch=args.epoch, eta_min=0.001)
        ),
        maximum_epoch=10, multiplier=batch_size / 96
    )
    optimizer_theta = skeleton.optim.ScheduledOptimzer(
        [p for n, p in model.named_parameters() if p.requires_grad and '.alpha' not in n],
        torch.optim.SGD,
        tag='theta',
        steps_per_epoch=len(train_loader),
        clip_grad_max_norm=5.0,
        lr=scheduler_theta,
        momentum=0.9, weight_decay=3e-4, nesterov=False
    )

    scheduler_alpha = skeleton.optim.gradual_warm_up(
        lambda lr: 3e-4,
        maximum_epoch=10, multiplier=batch_size / 96,
    )
    optimizer_alpha = skeleton.optim.ScheduledOptimzer(
        [p for n, p in model.named_parameters() if p.requires_grad and '.alpha' in n],
        torch.optim.Adam,
        tag='alpha',
        steps_per_epoch=len(train_loader),
        lr=scheduler_alpha,
        betas=(0.5, 0.999), weight_decay=1e-3
    )

    if args.gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.gpus)), output_device=0)
    model.register_forward_pre_hook(skeleton.nn.hooks.MoveToHook.get_forward_pre_hook(device=device, half=False))
    model.to(device).train()
    trainer = skeleton.darts.DartsTrainer(
        model,
        optimizers={
            'theta': optimizer_theta,
            'alpha': optimizer_alpha
        },
        metric_fns={
            'accuracy': skeleton.trainers.metrics.Accuracy(topk=1, scale=100.0)
        }
    )

    def initialize(module):
        if isinstance(module, torch.nn.BatchNorm2d):
            module.reset_running_stats()
        if isinstance(module, skeleton.nn.DropPath):
            module.drop_prob = 0.0
    model.apply(initialize)

    for epoch in range(args.epoch):
        model.apply_hard(cell=False, mixed=False)
        metrics_train_alpha, metrics_train_theta = trainer.train(train_loader, valid_loader, verbose=args.debug)
        metrics_valid = trainer.eval('valid', test_loader, verbose=args.debug)
        model.apply_hard(cell=True, mixed=True)
        metrics_test = trainer.eval('test', test_loader, verbose=args.debug)

        genotypes = []
        for name, genotype in model.genotypes().items():
            genotypes.append('[%s]' % name)
            genotypes += [str(path) for path in genotype['path']]
        genotypes_str = '\n'.join(genotypes)
        print(genotypes_str)
        skeleton.summary.text('train', 'genotypes', genotypes_str.replace('\n', '<BR/>').replace(' ', '&nbsp;'))

        writer.write(epoch)
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizers': {
                'theta': optimizer_theta.state_dict(),
                'alpha': optimizer_alpha.state_dict(),
            },
            'metrics': {
                'train': {
                    'alpha': metrics_train_alpha,
                    'theta': metrics_train_theta
                },
                'valid': metrics_valid,
                'test': metrics_test,
            }
        }, args.base_dir + '/models/epoch_%04d.pth' % epoch)


if __name__ == '__main__':
    parser = ConfigArgumentParser(conflict_handler='resolve')

    parser.add_argument('--num-class', type=int, default=10, help='10 or 100')
    parser.add_argument('--gpus', type=int, default=torch.cuda.device_count())

    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=50)

    parser.add_argument('--base-dir', type=str, default=None)
    parser.add_argument('--name', type=str, default=None)

    parser.add_argument('--log-filename', type=str, default='')
    parser.add_argument('--debug', action='store_true')
    parsed_args = parser.parse_args()

    parsed_args.gpus = 1

    log_format = '[%(asctime)s %(levelname)s] %(message)s'
    level = logging.DEBUG if parsed_args.debug else logging.INFO
    if not parsed_args.log_filename:
        logging.basicConfig(level=level, format=log_format, stream=sys.stderr)
    else:
        logging.basicConfig(level=level, format=log_format, filename=parsed_args.log_filename)

    name = '.'.join(parsed_args.config.split('/')[-1].split('.')[:-1])
    name += ('_' + parsed_args.name) if parsed_args.name is not None else ''
    if parsed_args.base_dir is None:
        parsed_args.base_dir = '/'.join([
            '.',
            'experiments',
            'cifar' + str(parsed_args.num_class),
            'darts',
            'NAS',
            name,
            datetime.datetime.now().strftime('%Y%m%d'),
            datetime.datetime.now().strftime('%H%M')
        ])

    if os.path.exists(parsed_args.base_dir):
        logging.warning('remove exists folder at %s', parsed_args.base_dir)
        shutil.rmtree(parsed_args.base_dir)
    os.makedirs(parsed_args.base_dir + '/models', exist_ok=True)

    main(parsed_args)

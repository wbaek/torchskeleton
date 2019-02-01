# -*- coding: utf-8 -*-
import os
import sys
import logging
import datetime
import random
import shutil
from collections import OrderedDict

import numpy as np
import torch
torch.backends.cudnn.benchmark = True

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
import skeleton


# Define Network Architecture
class BasicNet(skeleton.nn.modules.TraceModule):
    def __init__(self, num_classes=10):
        super(BasicNet, self).__init__()
        self.drop_path = skeleton.nn.DropPath(drop_prob=0.0)
        layers = [
            ('embed', torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(64),
                torch.nn.LeakyReLU(inplace=True),
            )),
            ('conv01', torch.nn.Sequential(
                skeleton.nn.Split(OrderedDict([
                    ('skip', skeleton.nn.FactorizedReduce(64, 128)),
                    ('deep', torch.nn.Sequential(
                        torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                        torch.nn.BatchNorm2d(128),
                        torch.nn.LeakyReLU(inplace=True),
                        torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                        torch.nn.BatchNorm2d(128),
                        self.drop_path,
                    ))
                ])),
                skeleton.nn.MergeSum(),
                torch.nn.LeakyReLU(inplace=True),
            )),
            ('conv02', torch.nn.Sequential(
                skeleton.nn.Split(OrderedDict([
                    ('skip', skeleton.nn.Identity()),
                    ('deep', torch.nn.Sequential(
                        torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                        torch.nn.BatchNorm2d(128),
                        torch.nn.LeakyReLU(inplace=True),
                        torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                        torch.nn.BatchNorm2d(128),
                        self.drop_path,
                    ))
                ])),
                skeleton.nn.MergeSum(),
                torch.nn.LeakyReLU(inplace=True),
            )),
            ('conv03', torch.nn.Sequential(
                skeleton.nn.Split(OrderedDict([
                    ('skip', skeleton.nn.FactorizedReduce(128, 256)),
                    ('deep', torch.nn.Sequential(
                        torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
                        torch.nn.BatchNorm2d(256),
                        torch.nn.LeakyReLU(inplace=True),
                        torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                        torch.nn.BatchNorm2d(256),
                        self.drop_path,
                    ))
                ])),
                skeleton.nn.MergeSum(),
                torch.nn.LeakyReLU(inplace=True),
            )),
            ('conv04', torch.nn.Sequential(
                skeleton.nn.Split(OrderedDict([
                    ('skip', skeleton.nn.Identity()),
                    ('deep', torch.nn.Sequential(
                        torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                        torch.nn.BatchNorm2d(256),
                        torch.nn.LeakyReLU(inplace=True),
                        torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                        torch.nn.BatchNorm2d(256),
                        self.drop_path,
                    ))
                ])),
                skeleton.nn.MergeSum(),
                torch.nn.LeakyReLU(inplace=True),
            )),

            ('global_pool', torch.nn.AdaptiveAvgPool2d((1, 1))),
            ('linear', torch.nn.Sequential(
                skeleton.nn.Flatten(),
                torch.nn.Linear(256, num_classes),
            ))
        ]
        self.layers = torch.nn.Sequential(OrderedDict(layers))
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets=None):  # pylint: disable=arguments-differ
        logits = self.layers(inputs)

        if targets is None:
            return logits, None
        loss = self.loss_fn(input=logits, target=targets)
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

    model = BasicNet(num_classes=args.num_class)
    model.to(device=device)

    # print Network Architecture
    print('---------- architecture ---------- ')
    model.register_trace_hooks()
    _ = model(
        inputs=torch.Tensor(*((2,) + data_shape[0][1:])).to(device),
        targets=torch.LongTensor(np.random.randint(0, 10, (2,))).to(device)
    )
    model.remove_trace_hooks()

    model_architecture = model.print_trace()
    skeleton.summary.text('train', 'architecture', model_architecture.replace('\n', '<BR/>').replace(' ', '&nbsp;'))
    writer.write(0)
    print('---------- done ---------- ')

    # lambda based optimizer scheduler
    scheduler = skeleton.optim.gradual_warm_up(
        skeleton.optim.get_discrete_epoch(skeleton.optim.get_cosine_schedule(0.025, args.epoch)),
        maximum_epoch=5, multiplier=32 / args.batch
    )
    optimizer = skeleton.optim.ScheduledOptimzer(
        [p for p in model.parameters() if p.requires_grad],
        torch.optim.SGD,
        steps_per_epoch=len(train_loader),
        lr=scheduler,
        momentum=0.9,
        weight_decay=3e-4,
        nesterov=True
    )

    if args.gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.gpus)), output_device=0)
    model.register_forward_pre_hook(skeleton.nn.hooks.MoveToHook.get_forward_pre_hook(device=device, half=False))
    model.to(device).train()
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

    # reset batchnorm running stats (to remove effect of warmup)
    def initialize(module):
        if isinstance(module, torch.nn.BatchNorm2d):
            module.reset_running_stats()
    model.apply(initialize)
    for epoch in range(1, args.epoch):
        # dynamic update member variables
        def apply_drop_prob(module):
            if isinstance(module, skeleton.nn.DropPath):
                drop_prob = 0.1 * epoch / args.epoch  # pylint: disable=cell-var-from-loop
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
    parser.add_argument('-c', '--num-class', type=int, default=10, help='10 or 100')
    parser.add_argument('-b', '--batch', type=int, default=256)
    parser.add_argument('-e', '--epoch', type=int, default=30)
    parser.add_argument('--gpus', type=int, default=torch.cuda.device_count())

    parser.add_argument('--base-dir', type=str, required=None)
    parser.add_argument('--name', type=str, required=None)

    parser.add_argument('--log-filename', type=str, default='')
    parser.add_argument('--debug', action='store_true')
    parsed_args = parser.parse_args()

    parsed_args.debug = True
    parsed_args.gpus = 1

    log_format = '[%(asctime)s %(levelname)s] %(message)s'
    level = logging.DEBUG if parsed_args.debug else logging.INFO
    if not parsed_args.log_filename:
        logging.basicConfig(level=level, format=log_format, stream=sys.stderr)
    else:
        logging.basicConfig(level=level, format=log_format, filename=parsed_args.log_filename)

    name = 'modified_resnet9'
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

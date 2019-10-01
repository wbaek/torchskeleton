# -*- coding: utf-8 -*-
import os
import sys
import shutil
import logging

import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter


base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)
import skeleton
import resnet


LOGGER = logging.getLogger(__name__)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="fast converge cifar10")

    parser.add_argument('--dataset-base', type=str, default='./data')
    parser.add_argument('--dataset-name', type=str, default='cifar10')
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--result-path', type=str, default=None)

    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--hypergrad', action='store_true')

    parser.add_argument('--download', action='store_true')
    parser.add_argument('--seed', type=lambda x: int(x, 0), default=None)

    parser.add_argument('--log-filename', type=str, default='')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def dataloaders(base, name, download, batch_size, device):
    num_class = 10 if name == 'cifar10' else 100
    dataset = torchvision.datasets.CIFAR10 if name == 'cifar10' else torchvision.datasets.CIFAR100

    train_dataset = dataset(
        root=base + '/cifar10',
        train=True,
        download=download
    )
    test_dataset = dataset(
        root=base + '/cifar10',
        train=False,
        download=download
    )

    post_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2471, 0.2435, 0.2616)
        ),
    ])

    train_dataloader = skeleton.data.FixedSizeDataLoader(
        skeleton.data.TransformDataset(
            skeleton.data.prefetch_dataset(
                skeleton.data.TransformDataset(
                    train_dataset,
                    transform=torchvision.transforms.Compose([
                        skeleton.data.transforms.Pad(4),
                        post_transform
                    ]),
                    index=0
                ),
                num_workers=16
            ),
            transform=torchvision.transforms.Compose([
                skeleton.data.transforms.TensorRandomCrop(32, 32),
                skeleton.data.transforms.TensorRandomHorizontalFlip(),
                skeleton.data.transforms.Cutout(8, 8)
            ]),
            index=0
        ),
        steps=None,  # for prefetch using infinit dataloader
        batch_size=batch_size,
        num_workers=32,
        pin_memory=False,
        drop_last=True,
        shuffle=True,
        # sampler=skeleton.data.StratifiedSampler(train_dataset.targets)
    )

    test_dataloader = torch.utils.data.DataLoader(
        skeleton.data.prefetch_dataset(
            skeleton.data.TransformDataset(
                test_dataset,
                transform=torchvision.transforms.Compose([
                    post_transform,
                ]),
                index=0
            ),
            num_workers=16
        ),
        batch_size=batch_size // 2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )

    train_dataloader = skeleton.data.PrefetchDataLoader(train_dataloader, device=device, half=False)
    test_dataloader = skeleton.data.PrefetchDataLoader(test_dataloader, device=device, half=False)
    return num_class, int(len(train_dataset) // batch_size), train_dataloader, test_dataloader


def main():
    timer = skeleton.utils.Timer()

    args = parse_args()
    log_format = '[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)03d] %(message)s'
    level = logging.DEBUG if args.debug else logging.INFO
    if not args.log_filename:
        logging.basicConfig(level=level, format=log_format, stream=sys.stderr)
    else:
        logging.basicConfig(level=level, format=log_format, filename=args.log_filename)

    torch.backends.cudnn.benchmark = True
    if args.seed is not None:
        skeleton.utils.set_random_seed_all(args.seed, deterministic=False)

    if args.result_path is not None:
        if os.path.exists(args.result_path):
            LOGGER.warning('remove exists result path at %s', args.result_path)
            shutil.rmtree(args.result_path)
        os.makedirs(args.result_path + '/models', exist_ok=True)
        writers = {
            'train': SummaryWriter(args.result_path + '/train'),
            'test': SummaryWriter(args.result_path + '/test'),
        }
    LOGGER.debug('args: %s', args)

    epochs = args.epoch
    batch_size = args.batch
    device = torch.device('cuda', 0)

    num_class, steps_per_epoch, train_loader, test_loader = dataloaders(
        args.dataset_base, args.dataset_name, args.download, batch_size, device
    )
    train_iter = iter(train_loader)

    model = resnet.resnet18(num_classes=num_class).to(device=device)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    metrics = skeleton.nn.Accuracy(1)

    optimizer = skeleton.optim.ScheduledOptimizer(
        [p for p in model.parameters() if p.requires_grad],
        skeleton.optim.HypergradSGD,
        steps_per_epoch=steps_per_epoch,
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4,
        # nesterov=True,
        hypergrad_lr=(1e-3 * batch_size / 128) if args.hypergrad else 0,
        hypergrad_momentum=0.9 if args.hypergrad else 0,
    )

    class ModelLoss(torch.nn.Module):
        def __init__(self, model, criterion):
            super(ModelLoss, self).__init__()
            self.model = model
            self.criterion = criterion

        def forward(self, inputs, targets=None):
            logits = self.model(inputs)
            if targets is not None:
                loss = self.criterion(logits, targets)
                return logits, loss
            return logits
    model = ModelLoss(model, criterion)

    # warmup
    torch.cuda.synchronize()
    model.train()
    for _ in range(2):
        inputs, targets = next(train_iter)
        logits, loss = model(inputs, targets)
        loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()
    timer('init')

    # train
    for epoch in range(epochs):
        model.train()
        train_metrics = {'loss': [], 'accuracy': []}
        timer('init', reset_step=True)
        for step in range(steps_per_epoch):
            inputs, targets = next(train_iter)
            logits, loss = model(inputs, targets)

            loss.sum().backward()
            accuracy = metrics(logits, targets)
            train_metrics['loss'].append(loss.detach())
            train_metrics['accuracy'].append(accuracy.detach())

            optimizer.update()
            optimizer.step()
            optimizer.zero_grad()
            LOGGER.debug(
                '[%02d] [%04d/%04d] [train] lr:%.5f loss:%.5f',
                epoch,
                step,
                steps_per_epoch,
                optimizer.get_learning_rate(),
                float(loss.detach())
            )
        timer('train')
        LOGGER.debug('lr:%s', ', '.join(['%.3f' % float(lr) for lr in optimizer.param_groups[0]['lr_per_layers']]))

        model.eval()
        test_metrics = {'loss': [], 'accuracy': []}
        with torch.no_grad():
            for inputs, targets in test_loader:
                logits, loss = model(inputs, targets)

                accuracy = metrics(logits, targets)

                test_metrics['loss'].append(loss.detach())
                test_metrics['accuracy'].append(accuracy.detach())
        timer('test')
        LOGGER.info(
            '[%02d] train loss:%.3f accuracy:%.3f test loss:%.3f accuracy:%.3f lr:%.5f',
            epoch,
            np.average([t.cpu().numpy() for t in train_metrics['loss']]),
            np.average([t.cpu().numpy() for t in train_metrics['accuracy']]),
            np.average([t.cpu().numpy() for t in test_metrics['loss']]),
            np.average([t.cpu().numpy() for t in test_metrics['accuracy']]),
            optimizer.get_learning_rate(),
        )

        if args.result_path is not None:
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, args.result_path + '/models/epoch%04d.pth' % (epoch))
            for key, value in train_metrics.items():
                writers['train'].add_scalar('metrics/{}'.format(key), np.average([t.cpu().numpy() for t in value]), global_step=epoch)
            for key, value in optimizer.params().items():
                if key == 'lr':
                    continue
                writers['train'].add_scalar('optimizer/{}'.format(key), value, global_step=epoch)
            writers['train'].add_scalar('optimizer/lr'.format(key), optimizer.get_learning_rate(), global_step=epoch)

            for key, value in test_metrics.items():
                writers['test'].add_scalar('metrics/{}'.format(key), np.average([t.cpu().numpy() for t in value]), global_step=epoch)


if __name__ == '__main__':
    main()

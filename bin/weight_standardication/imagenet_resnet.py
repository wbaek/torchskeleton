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
import resnet_gn_ws


LOGGER = logging.getLogger(__name__)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="fast converge imagenet")

    parser.add_argument('--dataset-base', type=str, default='./data')
    parser.add_argument('--seed', type=lambda x: int(x, 0), default=None)
    parser.add_argument('--result-path', type=str, default=None)

    parser.add_argument('--norm-layer', type=str, default='bn', help='[bn|gn|gn-ws]')

    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--num-gpus', type=int, default=torch.cuda.device_count())

    parser.add_argument('--log-filename', type=str, default='')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def dataloaders(base, input_size, batch_size, device, min_scale=0.08, num_workers=32):
    train_dataset = skeleton.data.ImageNet(root=base + '/imagenet', split='train')
    test_dataset = skeleton.data.ImageNet(root=base + '/imagenet',split='val')

    post_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    train_dataloader = skeleton.data.FixedSizeDataLoader(
        skeleton.data.TransformDataset(
            train_dataset,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(input_size, scale=(min_scale, 1.0)),
                torchvision.transforms.RandomHorizontalFlip(),
                post_transforms
            ]),
            index=0
        ),
        steps=None,  # for prefetch using infinit dataloader
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
        shuffle=True,
        # sampler=skeleton.data.StratifiedSampler(train_dataset.targets)
    )
    train_dataloader = skeleton.data.PrefetchDataLoader(train_dataloader, device=device)

    test_dataset = skeleton.data.TransformDataset(
        test_dataset,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(int(input_size * 1.14)),
            torchvision.transforms.CenterCrop(input_size),
            post_transforms
        ]),
        index=0
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False
    )
    return int(len(train_dataset) // batch_size), train_dataloader, test_dataloader


def main():
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

    environments = skeleton.utils.Environments()
    LOGGER.info('\n%s', environments)

    batch_size = args.batch * args.num_gpus
    device = torch.device('cuda', 0)

    steps_per_epoch, train_loader, test_loader = dataloaders(
        args.dataset_base, input_size=224, batch_size=batch_size, device=device
    )
    LOGGER.debug('prepared dataloader')

    init_lr = 0.1
    def schedule(epoch):
        if epoch < 30:
            return 1.0 * init_lr
        elif epoch < 60:
            return 0.1 * init_lr
        elif epoch < 80:
            return 0.01 * init_lr
        return 0.001 * init_lr
    batch_multiplier = batch_size / 256
    lr_scheduler = skeleton.optim.gradual_warm_up(
        skeleton.optim.get_lambda_scheduler(schedule),
        warm_up_epoch=5,
        multiplier=batch_multiplier
    )

    if args.norm_layer in ['bn', 'batchnorm']:
        model = torchvision.models.resnet.resnet101(pretrained=False)
    elif args.norm_layer in ['gn', 'groupnorm']:
        model = torchvision.models.resnet.resnet101(pretrained=False, norm_layer=resnet_gn_ws.Group32Norm)
    elif args.norm_layer in ['gn-ws', 'groupnorm-ws']:
        model = resnet_gn_ws.resnet101(pretrained=False)
    else:
        raise ValueError('invalid norm_layer arguments:%s', args.norm_layer)

    optimizer = skeleton.optim.ScheduledOptimizer(
        [p for p in model.parameters() if p.requires_grad],
        torch.optim.SGD,
        steps_per_epoch=steps_per_epoch,
        lr=lr_scheduler,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True
    )

    class Network(torch.nn.Module):
        def __init__(self, model):
            super(Network, self).__init__()
            self.model = model
            self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            self.metrics_fns = {
                'accuracy_top1': skeleton.nn.metrics.Accuracy(topk=1),
                'accuracy_top5': skeleton.nn.metrics.Accuracy(topk=5),
            }

        def forward(self, inputs, targets):
            logits = self.model(inputs)

            loss = self.criterion(logits, targets)
            metrics = {key: fn(logits, targets).detach() for key, fn, in self.metrics_fns.items()}
            metrics['loss'] = loss
            return metrics

    model = Network(model).to(device=device)
    if args.num_gpus > 1:
        parallel_model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpus)))
    else:
        parallel_model = model
    LOGGER.debug('initialize')

    train_iter = iter(train_loader)
    for epoch in range(90):
        metric_hist = []
        parallel_model.train()
        for step, (inputs, targets) in zip(range(steps_per_epoch), train_iter):
            metrics = parallel_model(inputs, targets)
            metrics['loss'].mean().backward()

            optimizer.step()
            optimizer.update()
            optimizer.zero_grad()

            metrics = {key: value for key, value in metrics.items()}
            metric_hist.append(metrics)
            LOGGER.debug('[train] [%03d] %04d/%04d lr:%.4f', epoch, step, steps_per_epoch, optimizer.get_learning_rate())
        train_metrics = {metric: np.average([float(m[metric].mean()) for m in metric_hist]) for metric in metric_hist[0].keys()}

        metric_hist = []
        with torch.no_grad():
            parallel_model.eval()
            for step, (inputs, targets) in enumerate(test_loader):
                metrics = parallel_model(inputs, targets)
                metrics = {key: value for key, value in metrics.items()}

                metric_hist.append(metrics)
                LOGGER.debug('[test] [%03d] %04d/%04d', epoch, step, len(test_loader))
        test_metrics = {metric: np.average([float(m[metric].mean()) for m in metric_hist]) for metric in metric_hist[0].keys()}

        LOGGER.info('[%03d] train:%s test:%s', epoch, train_metrics, test_metrics)
        if args.result_path is not None:
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, args.result_path + '/models/epoch%04d.pth' % (epoch))
            for key, value in train_metrics.items():
                writers['train'].add_scalar('metrics/{}'.format(key), value, global_step=epoch)
            for key, value in optimizer.params().items():
                writers['train'].add_scalar('optimizer/{}'.format(key), value, global_step=epoch)

            for key, value in test_metrics.items():
                writers['test'].add_scalar('metrics/{}'.format(key), value, global_step=epoch)


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
import os
import sys
import logging

import numpy as np
import torch
import torchvision


base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)
import skeleton


LOGGER = logging.getLogger(__name__)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="fast converge imagenet")

    parser.add_argument('--dataset-base', type=str, default='./data')
    parser.add_argument('--num-gpus', type=int, default=torch.cuda.device_count())
    parser.add_argument('--seed', type=lambda x: int(x, 0), default=None)

    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--batch', type=int, default=512)

    parser.add_argument('--log-filename', type=str, default='')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def dataloaders(base, input_size, batch_size, device, half=True, min_scale=0.08, test_prefetch=True, small=None, num_workers=32):
    # train_dataset = torchvision.datasets.ImageNet(
    train_dataset = skeleton.data.ImageNet(
        root=base + ('/imagenet' if small is None else '/imagenet-sz/{}'.format(small)),
        split='train'
    )
    # test_dataset = torchvision.datasets.ImageNet(
    test_dataset = skeleton.data.ImageNet(
        root=base + ('/imagenet' if small is None else '/imagenet-sz/{}'.format(small)),
        split='val'
    )

    post_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
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
        pin_memory=True,
        drop_last=True,
        shuffle=True,
        # sampler=skeleton.data.StratifiedSampler(train_dataset.targets)
    )

    test_dataset = skeleton.data.TransformDataset(
        test_dataset,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(int(input_size * 1.14)),
            # torchvision.transforms.CenterCrop(input_size),
            # post_transforms
            torchvision.transforms.TenCrop(input_size),
            torchvision.transforms.Lambda(
                lambda crops: torch.stack([
                    post_transforms(crop) for crop in crops
                ])
            )
        ]),
        index=0
    )
    if test_prefetch:
        test_dataset = skeleton.data.prefetch_dataset(
            test_dataset,
            num_workers=num_workers,
            device=device, half=half, contiguous=True
        )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size // 4,
        shuffle=False,
        num_workers=0 if test_prefetch else num_workers // 2,
        pin_memory=False,
        drop_last=False
    )

    train_dataloader = skeleton.data.PrefetchDataLoader(train_dataloader, device=device, half=half, contiguous=False)
    if not test_prefetch:
        test_dataloader = skeleton.data.PrefetchDataLoader(test_dataloader, device=device, half=half, contiguous=True)
    return int(len(train_dataset) // batch_size), train_dataloader, test_dataloader


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
    LOGGER.debug('args: %s', args)

    environments = skeleton.utils.Environments()
    LOGGER.debug('\n%s', environments)

    batch_size = args.batch * args.num_gpus
    device = torch.device('cuda', 0)
    half = not args.full

    datas = []
    steps_per_epoch, train_loader, test_loader = dataloaders(
        args.dataset_base, input_size=128, batch_size=batch_size * 2, small=160,
        device=device, half=half, test_prefetch=False
    )
    datas.append({
        'batch_size': batch_size * 2,
        'steps': steps_per_epoch,
        'train': iter(train_loader),
        'test': test_loader,
        'range': range(0, 35)
    })
    LOGGER.debug('dataloader step1')

    steps_per_epoch, train_loader, test_loader = dataloaders(
        args.dataset_base, input_size=224, batch_size=batch_size,
        device=device, half=half, test_prefetch=False
    )
    datas.append({
        'batch_size': batch_size,
        'steps': steps_per_epoch,
        'train': iter(train_loader),
        'test': test_loader,
        'range': range(35, 80)
    })
    LOGGER.debug('dataloader step2')

    steps_per_epoch, train_loader, test_loader = dataloaders(
        args.dataset_base, input_size=288, batch_size=batch_size // 2, min_scale=0.5,
        device=device, half=half, test_prefetch=False
    )
    datas.append({
        'batch_size': batch_size // 2,
        'steps': steps_per_epoch,
        'train': iter(train_loader),
        'test': test_loader,
        'range': range(80, 91)
    })
    LOGGER.debug('dataloader step3')

    lr = 0.1
    lr_multiplier = 10
    batch_multiplier = batch_size / 256
    epoch = 90
    lr_scheduler = skeleton.optim.gradual_warm_up(
        skeleton.optim.get_change_scale(
            skeleton.optim.get_discrete_epoch(
                skeleton.optim.get_piecewise(
                    [0,  4,  35, 37, 47,      57,       80,       82,       90],
                    [lr, lr, lr, lr, lr / 10, lr / 100, lr / 100, lr / 100, lr / 1000]
                )
            ),
            1.0 / (lr_multiplier * batch_size)
        ),
        warm_up_epoch=4,
        multiplier=lr_multiplier * batch_multiplier
    )

    if args.model == 'resnet50':
        model = torchvision.models.resnet.resnet50(pretrained=False)
    elif args.model == 'resnet101':
        model = torchvision.models.resnet.resnet101(pretrained=False)
    else:
        raise NotImplementedError

    model = model.to(device=device)
    if half:
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                pass
            else:
                module.half()

    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    metrics = skeleton.nn.Accuracy(5)

    optimizer = skeleton.optim.ScheduledOptimizer(
        [p for p in model.parameters() if p.requires_grad],
        torch.optim.SGD,
        steps_per_epoch=datas[0]['steps'],
        lr=lr_scheduler,
        momentum=0.9,
        weight_decay=1e-4 * batch_size,
        nesterov=True
    )

    class ModelLoss(torch.nn.Module):
        def __init__(self, model, criterion):
            super(ModelLoss, self).__init__()
            self.model = model
            self.criterion = criterion

        def forward(self, inputs, targets=None):
            logits = self.model(inputs)
            logits /= 2.0
            if targets is not None:
                loss = self.criterion(logits, targets)
                return logits, loss
            return logits

    model = ModelLoss(model, criterion)
    if args.num_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpus)))
    LOGGER.debug('init')

    # warmup
    torch.cuda.synchronize()
    model.train()
    # for data in datas:
    #     next(data['train'])
    for data in datas[:1]:
        for _ in range(2):
            inputs, targets = next(data['train'])
            logits, loss = model(inputs, targets)
            loss.sum().backward()
            model.zero_grad()
    torch.cuda.synchronize()
    LOGGER.debug('warmup')
    timer('init')

    # train
    results = ['epoch\thours\ttop5Accuracy']
    for e in range(epoch):
        idx = [idx for idx, data in enumerate(datas) if e in data['range']][0]
        for i in range(idx):
            keys = [key for key in datas[i].keys() if key != 'range']
            for k in keys: del datas[i][k]
        data = datas[idx]
        optimizer.steps_per_epoch = data['steps']
        current_batch_size = data['batch_size']

        model.train()
        train_loss_list = []
        timer('init', reset_step=True)
        for step in range(data['steps']):
            inputs, targets = next(data['train'])
            logits, loss = model(inputs, targets)

            loss = loss.sum()
            loss.backward()

            optimizer.update()
            optimizer.step()
            model.zero_grad()

            loss = loss.detach().cpu().numpy() / current_batch_size
            train_loss_list.append(loss)
            LOGGER.debug('[TRAIN] [%03d] [%03d/%03d] loss:%.3f lr:%.5f', e, step, data['steps'],
                         loss, optimizer.get_learning_rate() * batch_size,)
        timer('train')

        model.eval()
        count_list = []
        accuracy_list = []
        with torch.no_grad():
            for step, (inputs, targets) in enumerate(data['test']):
                current_batch_size = inputs.size(0)

                use_tta = len(inputs.size()) == 5
                if use_tta:
                    bs, ncrops, c, h, w = inputs.size()
                    inputs = inputs.view(-1, c, h, w)

                logits = model(inputs)

                if use_tta:
                    logits = logits.view(bs, ncrops, -1).mean(1)

                accuracy = metrics(logits, targets) * current_batch_size

                count_list.append(current_batch_size)
                accuracy_list.append(accuracy.detach().cpu().numpy())
                LOGGER.debug('[TEST] [%03d] [%03d/%03d] accuracyTop5:%.3f', e, step, len(data['test']),
                             accuracy / current_batch_size)
        timer('test')
        LOGGER.info(
            '[%02d] train loss:%.3f test accuracyTop5:%.3f lr:%.3f %s',
            e,
            np.average(train_loss_list),
            np.sum(accuracy_list) / np.sum(count_list),
            optimizer.get_learning_rate() * batch_size,
            timer
        )
        results.append('{epoch}\t{hour:.8f}\t{accuracy:.2f}'.format(**{
            'epoch': e,
            'hour': timer.accumulation['train'] / (60 * 60),
            'accuracy': float(np.sum(accuracy_list) / np.sum(count_list)) * 100.0
        }))
    print('\n'.join(results))
    torch.save(model.state_dict(), 'assets/kakaobrain_%s_single_imagenet.pth' % args.model)


if __name__ == '__main__':
    # > python bin/dawnbench/cifar10.py --seed 0xC0FFEE --download > log_dawnbench_cifar10.tsv
    main()

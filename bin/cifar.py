# -*- coding: utf-8 -*-
import os
import sys
import logging

import numpy as np
import torch
import torchvision


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
import skeleton


LOGGER = logging.getLogger(__name__)
torch.backends.cudnn.benchmark = True


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="fast converge cifar10")

    parser.add_argument('--dataset-base', type=str, default='./data')
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--epoch', type=int, default=24)

    parser.add_argument('--download', action='store_true')

    parser.add_argument('--log-filename', type=str, default='')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def dataloaders(base, download, batch_size, device):
    train_dataset = torchvision.datasets.CIFAR10(
        root=base + '/cifar10',
        train=True,
        download=download
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=base + '/cifar10',
        train=False,
        download=download
    )

    train_dataloader = skeleton.data.FixedSizeDataLoader(
        skeleton.data.TransformDataset(
            skeleton.data.prefetch_dataset(
                skeleton.data.TransformDataset(
                    train_dataset,
                    transform=torchvision.transforms.Compose([
                        skeleton.data.transforms.Pad(4),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(
                            mean=(0.4914, 0.4822, 0.4465),
                            std=(0.2471, 0.2435, 0.2616)
                        ),
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
        shuffle=True,
        num_workers=32,
        pin_memory=False,
        drop_last=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        skeleton.data.prefetch_dataset(
            skeleton.data.TransformDataset(
                test_dataset,
                transform=torchvision.transforms.Compose([
                    # skeleton.data.transforms.Pad(4),
                    # torchvision.transforms.ToPILImage(),
                    # torchvision.transforms.TenCrop((32, 32)),
                    # torchvision.transforms.Lambda(
                    #     lambda crops: torch.stack([
                    #         torchvision.transforms.Compose([
                    #             torchvision.transforms.ToTensor(),
                    #             torchvision.transforms.Normalize(
                    #                 mean=(0.4914, 0.4822, 0.4465),
                    #                 std=(0.2471, 0.2435, 0.2616)
                    #             )
                    #         ])(crop) for crop in crops
                    #     ])
                    # )
                    # torchvision.transforms.CenterCrop((28, 28)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=(0.4914, 0.4822, 0.4465),
                        std=(0.2471, 0.2435, 0.2616)
                    ),
                    torchvision.transforms.Lambda(
                        lambda tensor: torch.stack([
                            tensor, torch.flip(tensor, dims=[-1])
                        ], dim=0)
                    )
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

    train_dataloader = skeleton.data.PrefetchDataLoader(train_dataloader, device=device, half=True)
    test_dataloader = skeleton.data.PrefetchDataLoader(test_dataloader, device=device, half=True)
    return int(len(train_dataset) // batch_size), train_dataloader, test_dataloader


def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1, activation=True):
    return torch.nn.Sequential(
        torch.nn.Conv2d(channels_in, channels_out,
                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
        torch.nn.BatchNorm2d(channels_out),
        torch.nn.ReLU(True) if activation else torch.nn.Identity()
    )


def build_network(num_class=10):
    return torch.nn.Sequential(
        # torch.nn.Sequential(
        #     skeleton.nn.StrideConv2d(3, 128, kernel_size=3, padding=1),  # 16
        #     torch.nn.BatchNorm2d(128),
        #     torch.nn.ReLU(True),
        # ),dd
        conv_bn(3, 128, kernel_size=3, stride=1, padding=1),
        conv_bn(128, 128, kernel_size=3, stride=1, padding=1),
        torch.nn.MaxPool2d(2),

        skeleton.nn.Split(  # residual
            torch.nn.Identity(),
            torch.nn.Sequential(
                conv_bn(128, 128),
                conv_bn(128, 128),
                # skeleton.nn.SEBlock(128),
            )
        ),
        skeleton.nn.MergeSum(),
        # torch.nn.ReLU(inplace=True),

        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        torch.nn.MaxPool2d(2),  # 8

        # skeleton.nn.Split(  # residual
        #     torch.nn.Identity(),
        #     torch.nn.Sequential(
        #         conv_bn(256, 256),
        #         conv_bn(256, 256),
        #     )
        # ),
        # skeleton.nn.MergeSum(),

        conv_bn(256, 512, kernel_size=3, stride=1, padding=1),
        torch.nn.MaxPool2d(2),  # 4

        skeleton.nn.Split(  # residual
            torch.nn.Identity(),
            torch.nn.Sequential(
                conv_bn(512, 512),
                conv_bn(512, 512),
                # skeleton.nn.SEBlock(512),
            )
        ),
        skeleton.nn.MergeSum(),

        torch.nn.AdaptiveMaxPool2d((1, 1)),  # 1x1
        skeleton.nn.Flatten(),
        torch.nn.Linear(512, num_class, bias=False),
        skeleton.nn.Mul(0.125)
    )


def main():
    timer = skeleton.utils.Timer()

    args = parse_args()
    log_format = '[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)03d] %(message)s'
    level = logging.DEBUG if args.debug else logging.INFO
    if not args.log_filename:
        logging.basicConfig(level=level, format=log_format, stream=sys.stderr)
    else:
        logging.basicConfig(level=level, format=log_format, filename=args.log_filename)

    epoch = args.epoch
    batch_size = args.batch
    device = torch.device('cuda', 0)

    steps_per_epoch, train_loader, test_loader = dataloaders(args.dataset_base, args.download, batch_size, device)
    train_iter = iter(train_loader)
    steps_per_epoch = int(steps_per_epoch * 1.0)

    model = build_network().to(device=device)
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.weight.data.fill_(1.0)
        else:
            module.half()
    # print(model)

    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    metrics = skeleton.nn.Accuracy(1)

    lr_scheduler = skeleton.optim.get_change_scale(
        skeleton.optim.get_piecewise([0, 5, epoch], [0.1, 0.4, 0.001]),
        1.0 / batch_size
    )
    optimizer = skeleton.optim.ScheduledOptimizer(
        [p for p in model.parameters() if p.requires_grad],
        torch.optim.SGD,
        steps_per_epoch=steps_per_epoch,
        lr=lr_scheduler,
        momentum=0.9,
        weight_decay=5e-4 * batch_size,
        nesterov=True
    )

    class ModelLoss(torch.nn.Module):
        def __init__(self, model, criterion):
            super(ModelLoss, self).__init__()
            self.model = model
            self.criterion = criterion

        def forward(self, input, target):
            logits = self.model(input)
            loss = self.criterion(logits, target)
            return logits, loss
    # normal
    model = ModelLoss(model, criterion)
    # cutmix
    # model = skeleton.nn.CutMix(model, criterion, prob=0.5, beta=1.0)

    # warmup
    torch.cuda.synchronize()
    model.train()
    for _ in range(2):
        inputs, targets = next(train_iter)
        logits, loss = model(inputs, targets)
        loss.sum().backward()
        model.zero_grad()
    torch.cuda.synchronize()
    timer('init')

    # train
    for epoch in range(epoch):
        timer('init', reset_step=True)
        model.train()
        loss_list = []
        for step in range(steps_per_epoch):
            inputs, targets = next(train_iter)
            logits, loss = model(inputs, targets)

            loss.sum().backward()

            optimizer.update()
            optimizer.step()
            model.zero_grad()
            loss_list.append(loss.mean().detach().cpu().numpy())
        timer('train')

        model.eval()
        accuracy_list = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                origin_targets = targets
                use_tta = len(inputs.size()) == 5
                if use_tta:
                    bs, ncrops, c, h, w = inputs.size()
                    inputs = inputs.view(-1, c, h, w)
                    targets = torch.cat([targets for _ in range(ncrops)], dim=0)

                logits, loss = model(inputs, targets)
                if use_tta:
                    logits = logits.view(bs, ncrops, -1).mean(1)

                accuracy = metrics(logits, origin_targets)
                accuracy_list.append(accuracy.detach().cpu().numpy())
        timer('test')
        print(
            '[%02d] train loss:%.5f test accuracy:%5f lr:%.3f %s' % (
            epoch,
            np.average(loss_list),
            np.average(accuracy_list),
            optimizer.get_learning_rate() * batch_size,
            timer)
        )


if __name__ == '__main__':
    main()

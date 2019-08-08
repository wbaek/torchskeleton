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
    parser = argparse.ArgumentParser(description="fast infer cifar10")

    parser.add_argument('--dataset-base', type=str, default='./data')
    parser.add_argument('--model', type=str, required=True)

    parser.add_argument('--download', action='store_true')

    parser.add_argument('--log-filename', type=str, default='')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def dataloaders(base, download, device):
    test_dataset = torchvision.datasets.CIFAR10(
        root=base + '/cifar10',
        train=False,
        download=download
    )

    test_dataloader = skeleton.data.FixedSizeDataLoader(
        skeleton.data.prefetch_dataset(
            skeleton.data.TransformDataset(
                test_dataset,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.CenterCrop((30, 30)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=(0.4914, 0.4822, 0.4465),
                        std=(0.2471, 0.2435, 0.2616)
                    ),
                    torchvision.transforms.Lambda(
                        lambda tensor: torch.stack([
                            tensor, torch.flip(tensor, dims=[-1]),
                        ], dim=0)
                    )
                ]),
                index=0
            ),
            num_workers=16,
            device=device,
            half=True
        ),
        steps=None,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )

    # test_dataloader = skeleton.data.PrefetchDataLoader(test_dataloader, device=device, half=True)
    return len(test_dataset), test_dataloader, 2


def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1, bn=True, activation=True):
    op = [
            torch.nn.Conv2d(channels_in, channels_out,
                            kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
    ]
    if bn:
        op.append(torch.nn.BatchNorm2d(channels_out))
    if activation:
        op.append(torch.nn.ReLU(inplace=True))
    return torch.nn.Sequential(*op)


class Residual(torch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


def build_network(num_class=10):
    return torch.nn.Sequential(
        conv_bn(3, 64, kernel_size=3, stride=1, padding=1),  # 30
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),  # 15
        # torch.nn.MaxPool2d(2),

        Residual(torch.nn.Sequential(
            conv_bn(128, 128),
            conv_bn(128, 128),
        )),

        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),  # 15
        torch.nn.MaxPool2d(2),  # 8

        Residual(torch.nn.Sequential(
            conv_bn(256, 256),
            conv_bn(256, 256),
        )),

        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),  # 6

        torch.nn.AdaptiveMaxPool2d((1, 1)),
        skeleton.nn.Flatten(),
        torch.nn.Linear(128, num_class, bias=False),
        skeleton.nn.Mul(0.2)
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

    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda', 0)

    num_items, test_loader, batch_size = dataloaders(args.dataset_base, args.download, device)
    test_iter = iter(test_loader)

    model = build_network().to(device=device)
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            pass
        else:
            module.half()
    model.load_state_dict({key[6:]: value for key, value in torch.load(args.model).items()})
    model = model.eval()

    # JIT compile
    with torch.no_grad():
        example = torch.zeros(batch_size, 3, 30, 30, dtype=torch.float16).to(device=device)
        jit_model = torch.jit.trace(model, example)

    # warmup
    for _ in range(2):
        example = torch.zeros(batch_size, 3, 30, 30, dtype=torch.float16).to(device=device)
        jit_model(example)
    for _ in range(num_items):
        _ = next(test_iter)
    torch.cuda.synchronize()
    timer('init')

    # prediction
    predictions = []
    with torch.no_grad():
        for _ in range(num_items):
            inputs, _ = next(test_iter)
            logits = jit_model(inputs.view(batch_size, 3, 30, 30))
            predictions.append(logits.detach())
    torch.cuda.synchronize()
    timer('inference')

    predictions = torch.cat([p.view(1, batch_size, -1).mean(1) for p in predictions], dim=0)
    targets = torch.cat([next(test_iter)[1] for _ in range(num_items)], dim=0)
    accuracy = skeleton.nn.Accuracy(1)(predictions, targets)

    print('accuracy:%.2f%%, %.5fus per sample' % (
        accuracy * 100.0,
        (timer.accumulation['inference'] * 1000.0) / num_items
    ))

if __name__ == '__main__':
    # > python bin/dawnbench/cifar10_infer.py --model assets/kakaobrain_custom-resnet9_single_cifar10.pth
    main()

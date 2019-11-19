# -*- coding: utf-8 -*-
import os
import sys
import logging

import torch
import torchvision


base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)
import skeleton
import efficientnet


LOGGER = logging.getLogger(__name__)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="fast converge cifar10")

    parser.add_argument('-a', '--architecture', type=str, default='efficientnet-b0')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints/')
    parser.add_argument('--batch', type=int, default=None)

    parser.add_argument('--datapath', type=str, default='./data')
    parser.add_argument('--set', type=str, default='val')

    parser.add_argument('--num-gpus', type=int, default=torch.cuda.device_count())
    parser.add_argument('--log-filename', type=str, default='')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


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

    assert 'efficientnet' in args.architecture
    assert args.architecture.split('-')[1] in ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7']

    environments = skeleton.utils.Environments()
    device = torch.device('cuda', 0)

    if args.batch is None:
        args.batch = 128 if 'b0' in args.architecture else args.batch
        args.batch = 96 if 'b1' in args.architecture else args.batch
        args.batch = 64 if 'b2' in args.architecture else args.batch
        args.batch = 32 if 'b3' in args.architecture else args.batch
        args.batch = 16 if 'b4' in args.architecture else args.batch
        args.batch = 8 if 'b5' in args.architecture else args.batch
        args.batch = 6 if 'b6' in args.architecture else args.batch
        args.batch = 4 if 'b7' in args.architecture else args.batch

    input_size = efficientnet.EfficientNet.get_image_size(args.architecture)
    model = efficientnet.EfficientNet.from_pretrained(args.architecture, model_dir=args.checkpoint_dir).to(device=device)

    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    metric = skeleton.nn.AccuracyMany((1, 5))

    profiler = skeleton.nn.Profiler(model)
    params = profiler.params()
    flops = profiler.flops(torch.ones(1, 3, input_size, input_size, dtype=torch.float, device=device))

    LOGGER.info('args\n%s', args)
    LOGGER.info('environemtns\n%s', environments)
    LOGGER.info('arechitecture\n%s\ninput:%d\nprarms:%.2fM\nGFLOPs:%.3f', args.architecture, input_size, params / (1024 * 1024), flops / (1024 * 1024 * 1024))
    LOGGER.info('optimizers\nloss:%s', str(criterion))

    model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpus)))
    model = model.eval()
    batch = args.batch * args.num_gpus

    dataset = skeleton.data.ImageNet(
        root=args.datapath + '/imagenet',
        split=args.set,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(int(input_size * 1.14)),
            torchvision.transforms.CenterCrop(input_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch,
        shuffle=False, drop_last=False,
        num_workers=environments.num_cpus // 4 * args.num_gpus, pin_memory=False
    )
    dataloader = skeleton.data.PrefetchDataLoader(dataloader, device=device)

    inputs = torch.ones(batch, 3, input_size, input_size, dtype=torch.float, device=device)
    targets = torch.zeros(batch, dtype=torch.long, device=device)

    # warmup
    for _ in range(2):
        logits = model(inputs)
        loss = criterion(logits, targets)
        accuracies = metric(logits, targets)

    timer('init', reset_step=True, exclude_total=True)
    metrics = {
        'loss': torch.zeros(1, dtype=torch.float, device=device),
        'top1': torch.zeros(1, dtype=torch.float, device=device),
        'top5': torch.zeros(1, dtype=torch.float, device=device),
    }

    for step, (inputs, targets) in enumerate(dataloader):
        with torch.no_grad():
            timer('init', reset_step=True)
            inputs, targets = inputs.to(device=device), targets.to(device=device)

            logits = model(inputs)
            loss = criterion(logits, targets)
            accuracies = metric(logits, targets)
            timer('forward')

            metrics['loss'] += loss.detach()
            metrics['top1'] += accuracies[0].detach()
            metrics['top5'] += accuracies[1].detach()

            images = step * batch
            LOGGER.info('[%04d/%04d] throughput:%.4f images/sec', step, len(dataloader), images * timer.throughput())
            if step % 100 == 0:
                LOGGER.info('[%04d/%04d] accruacy:%.4f', step, len(dataloader), metrics['top1'].cpu().item() / (step+1))

    steps = len(dataloader)
    images = steps * batch
    LOGGER.info('throughput:%.4f images/sec loss:%.4f accuracy top1: %.4f accuracy top5: %.4f',
                images * timer.throughput(),
                metrics['loss'].cpu().item() / steps,
                metrics['top1'].cpu().item() / steps,
                metrics['top5'].cpu().item() / steps
                )

if __name__ == '__main__':
    # > python bin/benchmark/efficientnet.py
    main()

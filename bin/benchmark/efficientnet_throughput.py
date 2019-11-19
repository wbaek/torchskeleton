# -*- coding: utf-8 -*-
import os
import sys
import logging

import torch


base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)
import skeleton
import efficientnet


LOGGER = logging.getLogger(__name__)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="fast converge cifar10")

    parser.add_argument('-a', '--architecture', type=str, default='efficientnet-b0')
    parser.add_argument('--batch', type=int, default=None)
    parser.add_argument('--steps', type=int, default=50)

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
        args.batch *= 2

    input_size = efficientnet.EfficientNet.get_image_size(args.architecture)
    model = efficientnet.EfficientNet.from_name(args.architecture).to(device=device)

    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5, nesterov=True)

    profiler = skeleton.nn.Profiler(model)
    params = profiler.params()
    flops = profiler.flops(torch.ones(1, 3, input_size, input_size, dtype=torch.float, device=device))

    LOGGER.info('environemtns\n%s', environments)
    LOGGER.info('arechitecture\n%s\ninput:%d\nprarms:%.2fM\nGFLOPs:%.3f', args.architecture, input_size, params / (1024 * 1024), flops / (1024 * 1024 * 1024))
    LOGGER.info('optimizers\nloss:%s\noptimizer:%s', str(criterion), str(optimizer))
    LOGGER.info('args\n%s', args)

    model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpus)))
    batch = args.batch * args.num_gpus
    inputs = torch.ones(batch, 3, input_size, input_size, dtype=torch.float, device=device)
    targets = torch.zeros(batch, dtype=torch.long, device=device)

    # warmup
    for _ in range(2):
        logits = model(inputs)
        loss = criterion(logits, targets)

        loss.backward()
        model.zero_grad()

    timer('init', reset_step=True, exclude_total=True)
    for step in range(args.steps):
        timer('init', reset_step=True)

        logits = model(inputs)
        loss = criterion(logits, targets)
        timer('forward')

        loss.backward()
        timer('backward')

        optimizer.step()
        optimizer.zero_grad()
        timer('step')

        LOGGER.info('[%02d] %s', step, timer)

    images = args.steps * batch
    LOGGER.info('throughput:%.4f images/sec', images * timer.throughput())


if __name__ == '__main__':
    # > python bin/benchmark/efficientnet.py
    main()

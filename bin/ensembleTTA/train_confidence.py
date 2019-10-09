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


LOGGER = logging.getLogger(__name__)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="imagenet ensemle test time augmentation")

    parser.add_argument('--dataset-base', type=str, default='./data')
    parser.add_argument('--seed', type=lambda x: int(x, 0), default=None)
    parser.add_argument('--result-path', type=str, default=None)
    parser.add_argument('--subset', type=int, default=None)

    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--init-lr', type=float, default=0.005)

    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--epoch', type=int, default=90)
    parser.add_argument('--num-gpus', type=int, default=torch.cuda.device_count())

    parser.add_argument('--log-filename', type=str, default='')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def dataloaders(base, input_size, batch_size, device, min_scale=0.08, num_workers=32, subset=None):
    train_dataset = skeleton.data.ImageNet(root=base + '/imagenet', split='train')
    test_dataset = skeleton.data.ImageNet(root=base + '/imagenet',split='val')

    if subset is not None and subset > 1:
        random_indices = torch.randperm(len(train_dataset))
        train_dataset = torch.utils.data.Subset(train_dataset, indices=random_indices[:len(train_dataset) // subset])
        random_indices = torch.randperm(len(test_dataset))
        test_dataset = torch.utils.data.Subset(test_dataset, indices=random_indices[:len(test_dataset) // subset])

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

    # test time agumentation candidates
    transforms = [
        torchvision.transforms.Compose([
            torchvision.transforms.Resize(int(input_size * scale)),
            torchvision.transforms.CenterCrop(input_size),
            post_transforms
        ]) for scale in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    ]

    test_dataset = skeleton.data.TransformDataset(
        test_dataset,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Lambda(
                lambda x: torch.stack([
                    t(x) for t in transforms
                ])
            )
        ]),
        index=0
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size // 16,
        shuffle=False,
        num_workers=num_workers // 2,
        pin_memory=False,
        drop_last=False
    )

    train_dataloader = skeleton.data.PrefetchDataLoader(train_dataloader, device=device)
    test_dataloader = skeleton.data.PrefetchDataLoader(test_dataloader, device=device)
    return int(len(train_dataset) // batch_size), train_dataloader, test_dataloader


class ConfidenceNetwork(torch.nn.Module):
    def __init__(self, model):
        super(ConfidenceNetwork, self).__init__()
        self.model = model

        in_features = self.model.fc.in_features
        num_features = in_features // 4
        self.confidence = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=num_features, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=num_features, out_features=1),
        )

        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, x):
        with torch.no_grad():
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)

            x = self.model.avgpool(x)
            x = x.reshape(x.size(0), -1)

            logits = self.model.fc(x)

        confidences = self.confidence(x.detach())
        return confidences, logits

    def loss(self, confidences, logits, targets, tau=1.0):
        predictions = torch.softmax(logits / tau, dim=1)
        targets = predictions * torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1)
        targets = torch.sum(targets, dim=-1, keepdim=True)
        return self.loss_fn(confidences, targets)


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
        args.dataset_base, input_size=224, batch_size=batch_size, device=device, subset=args.subset
    )
    LOGGER.debug('prepared dataloader')

    init_lr = args.init_lr
    batch_multiplier = batch_size / 256
    lr_scheduler = skeleton.optim.gradual_warm_up(
        skeleton.optim.get_cosine_scheduler(init_lr, maximum_epoch=args.epoch),
        warm_up_epoch=5,
        multiplier=batch_multiplier
    )

    base_model = torchvision.models.resnet50(pretrained=True)
    # base_model = torchvision.models.resnet101(pretrained=True)
    # base_model = torchvision.models.resnext101_32x8d(pretrained=True)

    model = ConfidenceNetwork(base_model)

    optimizer = skeleton.optim.ScheduledOptimizer(
        [p for p in model.confidence.parameters() if p.requires_grad],
        torch.optim.SGD,
        steps_per_epoch=steps_per_epoch,
        lr=lr_scheduler,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True
    )

    metrics_fns = {
        'accuracy_top1': skeleton.nn.metrics.Accuracy(topk=1),
        'accuracy_top5': skeleton.nn.metrics.Accuracy(topk=5),
    }

    model = model.to(device=device)
    if args.num_gpus > 1:
        parallel_model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpus)))
    else:
        parallel_model = model
    LOGGER.debug('initialize')

    metric_hist = []
    with torch.no_grad():
        parallel_model.eval()
        for step, (inputs, targets) in enumerate(test_loader):
            bs, num_transforms, c, h, w = inputs.size()
            inputs = inputs.view(-1, c, h, w)

            confidences, logits = parallel_model(inputs)

            logits = logits.view(bs, num_transforms, -1)

            # # weighted sum based on confidences
            # confidences = confidences.view(bs, num_transforms, 1)
            # confidences = torch.sigmoid(confidences)
            # logits = logits * confidences
            # logits = logits.sum(dim=1)

            # select max
            confidences = confidences.view(bs, num_transforms)
            idx = torch.argmax(confidences, dim=1)
            logits = logits[np.arange(bs), idx.view(-1), :]

            metrics = {key: fn(logits, targets).detach() for key, fn, in metrics_fns.items()}
            metric_hist.append(metrics)
            LOGGER.debug('[test] [%03d] %04d/%04d', 0, step, len(test_loader))
    test_metrics = {metric: np.average([float(m[metric].mean()) for m in metric_hist]) for metric in metric_hist[0].keys()}
    LOGGER.info('[%03d] test:%s', 0, test_metrics)

    train_iter = iter(train_loader)
    for epoch in range(args.epoch):
        metric_hist = []
        parallel_model.train()
        for step, (inputs, targets) in zip(range(steps_per_epoch), train_iter):
            confidences, logits = parallel_model(inputs)

            metrics = {key: fn(logits, targets).detach() for key, fn, in metrics_fns.items()}

            loss = model.loss(confidences, logits, targets, tau=args.tau)
            loss.backward()

            optimizer.step()
            optimizer.update()
            optimizer.zero_grad()

            metrics = {key: value for key, value in metrics.items()}
            metrics['loss'] = loss.detach()
            metric_hist.append(metrics)
            LOGGER.debug('[train] [%03d] %04d/%04d lr:%.4f', epoch, step, steps_per_epoch, optimizer.get_learning_rate())

            if step % 10 == 0:
                train_metrics = {metric: np.average([float(m[metric].mean()) for m in metric_hist]) for metric in metric_hist[0].keys()}
                LOGGER.debug('[%03d] train:%s', epoch, train_metrics)
        train_metrics = {metric: np.average([float(m[metric].mean()) for m in metric_hist]) for metric in metric_hist[0].keys()}
        LOGGER.debug('[%03d] train:%s', epoch, train_metrics)

        metric_hist = []
        with torch.no_grad():
            parallel_model.eval()
            for step, (inputs, targets) in enumerate(test_loader):
                bs, num_transforms, c, h, w = inputs.size()
                inputs = inputs.view(-1, c, h, w)

                confidences, logits = parallel_model(inputs)

                logits = logits.view(bs, num_transforms, -1)

                # # weighted sum based on confidences
                # confidences = confidences.view(bs, num_transforms, 1)
                # confidences = torch.sigmoid(confidences)
                # logits = logits * confidences
                # logits = logits.sum(dim=1)

                # select max
                confidences = confidences.view(bs, num_transforms)
                idx = torch.argmax(confidences, dim=1)
                logits = logits[np.arange(bs), idx.view(-1), :]

                metrics = {key: fn(logits, targets).detach() for key, fn, in metrics_fns.items()}
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

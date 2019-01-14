# -*- coding: utf-8 -*-
import logging

import skorch
import numpy as np
import torch
import torchvision as tv

LOGGER = logging.getLogger(__name__)

CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]


class Cifar:
    @staticmethod
    def sets(batch_size, num_classes=10, root='./data'):
        assert num_classes in [10, 100]

        dataset = tv.datasets.CIFAR10 if num_classes == 10 else tv.datasets.CIFAR100
        data_shape = [(batch_size, 3, 32, 32), (batch_size, 1)]

        transform_train = tv.transforms.Compose([
            tv.transforms.Pad(4, padding_mode='reflect'),
            tv.transforms.RandomCrop(32),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        transform_valid = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        train_set = dataset(root=root, train=True, download=True, transform=transform_train)
        test_set = dataset(root=root, train=False, download=True, transform=transform_valid)

        return train_set, test_set, data_shape

    @staticmethod
    def loader(batch_size, num_classes=10, root='./data', num_workers=8):
        train_set, test_set, data_shape = Cifar.sets(batch_size, num_classes=num_classes, root=root)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)

        return train_loader, test_loader, data_shape


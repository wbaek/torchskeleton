# -*- coding: utf-8 -*-
from skeleton.datasets.cifar import Cifar


def test_cifar_dataset():
    train_set, valid_set, data_shape = Cifar.sets(1, num_classes=10)
    assert data_shape == [(1, 3, 32, 32), (1, 1)]

    train_set, valid_set, data_shape = Cifar.sets(1, num_classes=100)
    assert data_shape == [(1, 3, 32, 32), (1, 1)]

    train_loader, valid_loader, data_shape = Cifar.loader(1, num_classes=10)
    assert len(train_loader) == 50000
    assert len(valid_loader) == 10000

    inputs, targets = next(iter(train_loader))
    assert (inputs.shape) == data_shape[0]
    assert (targets.shape) == data_shape[1]

    inputs, targets = next(iter(valid_loader))
    assert (inputs.shape) == data_shape[0]
    assert (targets.shape) == data_shape[1]

    train_loader, valid_loader, data_shape = Cifar.loader(1, num_classes=100)
    assert len(train_loader) == 50000
    assert len(valid_loader) == 10000

    inputs, targets = next(iter(train_loader))
    assert (inputs.shape) == data_shape[0]
    assert (targets.shape) == data_shape[1]

    inputs, targets = next(iter(valid_loader))
    assert (inputs.shape) == data_shape[0]
    assert (targets.shape) == data_shape[1]

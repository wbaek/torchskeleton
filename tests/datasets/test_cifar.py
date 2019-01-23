# -*- coding: utf-8 -*-
from skeleton.datasets.cifar import Cifar


def test_cifar_dataset():
    train_set, test_set, data_shape = Cifar.sets(1, num_classes=10)
    assert data_shape == [(1, 3, 32, 32), (1,)]

    train_set, test_set, data_shape = Cifar.sets(1, num_classes=100)
    assert data_shape == [(1, 3, 32, 32), (1,)]

    train_loader, test_loader, data_shape = Cifar.loader(1, num_classes=10)
    assert len(train_loader) == 50000
    assert len(test_loader) == 10000

    inputs, targets = next(iter(train_loader))
    assert (inputs.shape) == data_shape[0]
    assert (targets.shape) == data_shape[1]

    inputs, targets = next(iter(test_loader))
    assert (inputs.shape) == data_shape[0]
    assert (targets.shape) == data_shape[1]

    train_loader, test_loader, data_shape = Cifar.loader(1, num_classes=100)
    assert len(train_loader) == 50000
    assert len(test_loader) == 10000

    inputs, targets = next(iter(train_loader))
    assert (inputs.shape) == data_shape[0]
    assert (targets.shape) == data_shape[1]

    inputs, targets = next(iter(test_loader))
    assert (inputs.shape) == data_shape[0]
    assert (targets.shape) == data_shape[1]


def test_cifar_dataset_cv_split():
    train_loader, valid_loader, test_loader, data_shape = Cifar.loader(1, num_classes=10, cv_ratio=0.1)
    assert len(train_loader) == 45000
    assert len(valid_loader) == 5000
    assert len(test_loader) == 10000

    train_loader, valid_loader, test_loader, data_shape = Cifar.loader(1, num_classes=10, cv_ratio=0.2)
    assert len(train_loader) == 40000
    assert len(valid_loader) == 10000
    assert len(test_loader) == 10000

    train_loader, valid_loader, test_loader, data_shape = Cifar.loader(1, num_classes=10, cv_ratio=0.5)
    assert len(train_loader) == 25000
    assert len(valid_loader) == 25000
    assert len(test_loader) == 10000

    train_loader, valid_loader, test_loader, data_shape = Cifar.loader(1, num_classes=10, cv_ratio=0.9)
    assert len(train_loader) == 5000
    assert len(valid_loader) == 45000
    assert len(test_loader) == 10000

    train_loader, valid_loader, test_loader, data_shape = Cifar.loader(32, num_classes=10, cv_ratio=0.1)
    assert len(train_loader) == 45000 // 32
    assert len(valid_loader) == 5000 // 32 + 1
    assert len(test_loader) == 10000 // 32 + 1

# -*- coding: utf-8 -*-
from skeleton.datasets.imagenet import Imagenet


def test_imagenet_dataset():
    train_set, valid_set, data_shape = Imagenet.sets(1)
    assert data_shape == [(1, 3, 224, 224), (1,)]

    train_loader, valid_loader, data_shape = Imagenet.loader(1)
    assert len(train_loader) == 50000
    assert len(valid_loader) == 10000

    inputs, targets = next(iter(train_loader))
    assert (inputs.shape) == data_shape[0]
    assert (targets.shape) == data_shape[1]

    inputs, targets = next(iter(valid_loader))
    assert (inputs.shape) == data_shape[0]
    assert (targets.shape) == data_shape[1]

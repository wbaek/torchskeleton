# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch.nn import functional

from skeleton.nn.modules.wrappers import Identity, Mul, Add, Concat, Flatten, GlobalPool


def test_identity():
    x = np.random.rand(128, 3, 32, 32).astype(np.float32)
    y = Identity()(torch.Tensor(x))
    assert x.shape == y.shape
    assert (x == y.numpy()).all()


def test_mul():
    x = np.random.rand(128, 3, 32, 32).astype(np.float32)
    y = Mul(0.5)(torch.Tensor(x))
    assert x.shape == y.shape
    assert (x * 0.5 == y.numpy()).all()

    y = Mul(2.0)(torch.Tensor(x))
    assert x.shape == y.shape
    assert (x * 2.0 == y.numpy()).all()


def test_add():
    x1 = np.random.rand(128, 3, 32, 32).astype(np.float32)
    x2 = np.random.rand(128, 3, 32, 32).astype(np.float32)
    y = Add()(torch.Tensor(x1), torch.Tensor(x2))
    assert x1.shape == y.shape
    assert x2.shape == y.shape
    assert (x1 + x2 == y.numpy()).all()


def test_concat():
    x1 = np.random.rand(128, 3, 32, 32).astype(np.float32)
    x2 = np.random.rand(128, 3, 32, 32).astype(np.float32)
    y = Concat()(torch.Tensor(x1), torch.Tensor(x2))
    assert y.shape == (128, 6, 32, 32)

    y = Concat(dim=2)(torch.Tensor(x1), torch.Tensor(x2))
    assert y.shape == (128, 3, 64, 32)

    y = Concat(dim=3)(torch.Tensor(x1), torch.Tensor(x2))
    assert y.shape == (128, 3, 32, 64)


def test_flatten():
    x = np.random.rand(128, 3, 32, 32).astype(np.float32)
    y = Flatten()(torch.Tensor(x))
    assert y.shape == (128, 3 * 32 * 32)


def test_global_pool():
    x = np.random.rand(128, 3, 32, 32).astype(np.float32)
    y = GlobalPool()(torch.Tensor(x))
    assert y.shape == (128, 3, 1, 1)

    y = GlobalPool(functional.adaptive_max_pool2d)(torch.Tensor(x))
    assert y.shape == (128, 3, 1, 1)

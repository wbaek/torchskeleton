# -*- coding: utf-8 -*-
import torch
import numpy as np

from skeleton.nn.modules.wrappers import Identity, Mul, Flatten, Concat, MergeSum, Split


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


def test_flatten():
    x = np.random.rand(128, 3, 32, 32).astype(np.float32)
    y = Flatten()(torch.Tensor(x))
    assert y.shape == (128, 3 * 32 * 32)


def test_concat():
    x1 = np.random.rand(128, 3, 32, 32).astype(np.float32)
    x2 = np.random.rand(128, 3, 32, 32).astype(np.float32)
    y = Concat()(torch.Tensor(x1), torch.Tensor(x2))
    assert y.shape == (128, 6, 32, 32)

    y = Concat(dim=2)(torch.Tensor(x1), torch.Tensor(x2))
    assert y.shape == (128, 3, 64, 32)

    y = Concat(dim=3)(torch.Tensor(x1), torch.Tensor(x2))
    assert y.shape == (128, 3, 32, 64)


def test_merge_sum():
    x1 = np.random.rand(128, 3, 32, 32).astype(np.float32)
    x2 = np.random.rand(128, 3, 32, 32).astype(np.float32)
    y = MergeSum()(torch.Tensor(x1), torch.Tensor(x2))
    assert y.shape == (128, 3, 32, 32)
    assert (x1 + x2 == y.numpy()).all()

    y = MergeSum(torch.Tensor([1, 2]))(torch.Tensor(x1), torch.Tensor(x2))
    assert (x1 + (x2 * 2) == y.numpy()).all()
    y = MergeSum(torch.Tensor([2, 1]))(torch.Tensor(x1), torch.Tensor(x2))
    assert ((x1 * 2) + x2 == y.numpy()).all()

    y = MergeSum([1, 2])(torch.Tensor(x1), torch.Tensor(x2))
    assert (x1 + (x2 * 2) == y.numpy()).all()
    y = MergeSum([2, 1])(torch.Tensor(x1), torch.Tensor(x2))
    assert ((x1 * 2) + x2 == y.numpy()).all()


def test_split():
    x = np.random.rand(128, 3, 32, 32).astype(np.float32)
    m = Split(torch.nn.Sequential(), Identity(), Mul(0.5), Mul(1.5))
    y1, y2, y3, y4 = m(torch.Tensor(x))

    assert y1.shape == (128, 3, 32, 32)
    assert y2.shape == (128, 3, 32, 32)
    assert y3.shape == (128, 3, 32, 32)
    assert y4.shape == (128, 3, 32, 32)

    assert (x == y1.numpy()).all()
    assert (x == y2.numpy()).all()
    assert (x * 0.5 == y3.numpy()).all()
    assert (x * 1.5 == y4.numpy()).all()

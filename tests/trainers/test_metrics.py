# -*- coding: utf-8 -*-
import pytest
import torch
import numpy as np

from skeleton.trainers.metrics import Accuracy, AccuracyMany

def test_accuracy_many():
    accuracy = AccuracyMany(topk=(1, 5, 10))
    output = torch.from_numpy(np.array([
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
    ]))
    target = torch.from_numpy(np.array([
        [0], [1], [2], [3], [4], [5], [6], [7], [8], [9]
    ], dtype=np.int64))
    res = [float(r.numpy()) for r in accuracy(output, target)]
    assert res[0] == pytest.approx(0.1)
    assert res[1] == pytest.approx(0.5)
    assert res[2] == pytest.approx(1.0)

    target = torch.from_numpy(np.array([
        [0], [1], [2], [3], [4], [5], [3], [2], [1], [0]
    ], dtype=np.int64))
    res = [float(r.numpy()) for r in accuracy(output, target)]
    assert res[0] == pytest.approx(0.2)
    assert res[1] == pytest.approx(0.9)
    assert res[2] == pytest.approx(1.0)


def test_accuracy():
    accuracy = Accuracy(topk=1)
    output = torch.from_numpy(np.array([
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
    ]))
    target = torch.from_numpy(np.array([
        [0], [1], [2], [3], [4], [5], [6], [7], [8], [9]
    ], dtype=np.int64))
    res = accuracy(output, target)
    assert float(res) == pytest.approx(0.1)

    target = torch.from_numpy(np.array([
        [0], [1], [2], [3], [4], [5], [3], [2], [1], [0]
    ], dtype=np.int64))
    res = accuracy(output, target)
    assert float(res) == pytest.approx(0.2)

    accuracy = Accuracy(topk=5)
    target = torch.from_numpy(np.array([
        [0], [1], [2], [3], [4], [5], [6], [7], [8], [9]
    ], dtype=np.int64))
    res = accuracy(output, target)
    assert float(res) == pytest.approx(0.5)

    target = torch.from_numpy(np.array([
        [0], [1], [2], [3], [4], [5], [3], [2], [1], [0]
    ], dtype=np.int64))
    res = accuracy(output, target)
    assert float(res) == pytest.approx(0.9)

    accuracy = Accuracy(topk=10)
    target = torch.from_numpy(np.array([
        [0], [1], [2], [3], [4], [5], [6], [7], [8], [9]
    ], dtype=np.int64))
    res = accuracy(output, target)
    assert float(res) == pytest.approx(1.0)

    target = torch.from_numpy(np.array([
        [0], [1], [2], [3], [4], [5], [3], [2], [1], [0]
    ], dtype=np.int64))
    res = accuracy(output, target)
    assert float(res) == pytest.approx(1.0)

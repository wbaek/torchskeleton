# -*- coding: utf-8 -*-
import os
import pytest
import torch
import numpy as np

from skeleton.nn.modules.module import IOModule, MoveToModule


FIXTURE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


@pytest.mark.filterwarnings("ignore:MarkInfo")
@pytest.mark.datafiles(
    os.path.join(FIXTURE_DIR, 'datas', 'dummy.pth')
)
def test_save_and_load(datafiles):
    filenames = [str(f) for f in datafiles.listdir()]

    class BasicModel(IOModule):  # pylint: disable=abstract-method
        def __init__(self):
            super(BasicModel, self).__init__()
            self.layers = torch.nn.ModuleList([
                torch.nn.Linear(10, 1),
                torch.nn.ReLU(inplace=True)
            ])

        def reset_parameters(self):
            torch.nn.init.kaiming_normal_(self.layers[0].weight, 10, mode='fan_in', nonlinearity='relu')

    model = BasicModel()
    model.reset_parameters()
    original_weights = [p.detach().numpy().copy() for p in model.parameters()]

    model.save(filenames[0])

    model.reset_parameters()
    new_weights = [p.detach().numpy().copy() for p in model.parameters()]

    model.load(filenames[0])
    loaded_weights = [p.detach().numpy().copy() for p in model.parameters()]

    assert sum([np.linalg.norm(p1 - p2) for p1, p2 in zip(original_weights, new_weights)]) > 1e-5
    assert sum([np.linalg.norm(p1 - p2) for p1, p2 in zip(original_weights, loaded_weights)]) < 1e-5


def test_move_to_module():
    class BasicModel(MoveToModule):  # pylint: disable=abstract-method
        def forward(self, x):
            return x

    model = BasicModel()
    model.half()

    x = torch.Tensor(10, 10)
    assert x.dtype == torch.float32

    y = model(x)
    assert x.dtype == torch.float16
    assert y.dtype == torch.float16

# -*- coding: utf-8 -*-
from __future__ import absolute_import
import copy
import logging

import torch
from torch.utils.data import Dataset


LOGGER = logging.getLogger(__name__)


class Copy(Dataset):
    def __init__(self, dataset, index=0):
        self.dataset = dataset
        self.index = index

    def __getitem__(self, index):
        tensors = self.dataset[index]
        tensors = list(tensors)

        img = tensors[self.index]

        tensors.insert(self.index+1, copy.deepcopy(img))
        return tuple(tensors)

    def __len__(self):
        return len(self.dataset)


class TransformDataset(Dataset):
    def __init__(self, dataset, transform=None, index=None):
        self.dataset = dataset
        self.transform = transform
        self.index = index

    def __getitem__(self, index):
        tensors = self.dataset[index]
        tensors = list(tensors)

        if self.transform is not None:
            if self.index is None:
                tensors = self.transform(*tensors)
            elif isinstance(self.index, int):
                tensors[self.index] = self.transform(tensors[self.index])
            elif isinstance(self.index, list):
                for idx in self.index:
                    tensors[idx] = self.transform(tensors[idx])

        return tuple(tensors)

    def __len__(self):
        return len(self.dataset)


def prefetch_dataset(dataset, num_workers=4, batch_size=32, device=None, half=False, contiguous=False):
    if isinstance(dataset, list) and isinstance(dataset[0], torch.Tensor):
        tensors = dataset
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False, drop_last=False,
            num_workers=num_workers, pin_memory=False
        )
        tensors = [t for t in dataloader]
        tensors = [torch.cat(t, dim=0) for t in zip(*tensors)]

    if device is not None:
        tensors = [t.to(device=device) for t in tensors]
    if half:
        tensors = [t.half() if t.is_floating_point() else t for t in tensors]
    if contiguous:
        tensors = [t.contiguous() for t in tensors]

    return torch.utils.data.TensorDataset(*tensors)

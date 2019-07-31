# -*- coding: utf-8 -*-
# pylint: disable=wildcard-import
import torch
from sklearn.model_selection import StratifiedShuffleSplit


def split(dataset, labels, cv_ratio, cv_index=0, seed=0xC0FFEE):
    sss = StratifiedShuffleSplit(n_splits=cv_index + 1, test_size=cv_ratio, random_state=seed)
    sss = sss.split(list(range(len(labels))), labels)

    for _ in range(cv_index + 1):
        train_idx, valid_idx = next(sss)

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    valid_dataset = torch.utils.data.Subset(dataset, valid_idx)
    return train_dataset, valid_dataset

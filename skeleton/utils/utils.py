# -*- coding: utf-8 -*-
from __future__ import absolute_import
import os
import random
import platform
import multiprocessing
from typing import NamedTuple

import numpy as np
import torch
import torchvision


def set_random_seed_all(seed, deterministic=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

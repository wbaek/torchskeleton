# -*- coding: utf-8 -*-
from __future__ import absolute_import
import os
import shutil
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


def _to_str_cude_device(properties):
    return '{0.name} version:{0.major}.{0.minor} mpcount:{0.multi_processor_count} memory:{0.total_memory}'.format(properties)


class Environments(NamedTuple):
    platform_name: str = platform.system() + ' ' + platform.release() + ' ' + platform.machine()
    python_version: str = str(platform.python_version())
    torch_version: str = str(torch.__version__)
    torchvision_version: str = str(torchvision.__version__)
    num_cpus: int = multiprocessing.cpu_count()
    num_gpus: int = 0 if not torch.cuda.is_available() else torch.cuda.device_count()
    gpu_device: str = '' if not torch.cuda.is_available() else _to_str_cude_device(torch.cuda.get_device_properties(0))
    cuda_version: str = '' if not torch.cuda.is_available() else torch.version.cuda
    cudnn_version: int = 0 if not torch.cuda.is_available() else torch.backends.cudnn.version()

    def __str__(self):
        return 'platform_name: {0.platform_name}\n' \
               'python_version: {0.python_version}\n' \
               'torch_version: {0.torch_version}\n' \
               'torchvision_version: {0.torchvision_version}\n' \
               'num_cpus: {0.num_cpus}\n' \
               'num_gpus: {0.num_gpus}\n' \
               'gpu_device: {0.gpu_device}\n' \
               'cuda_version: {0.cuda_version}\n' \
               'cudnn_version: {0.cudnn_version}'.format(self)


def save_checkpoint(path, state, is_best=False, filename='checkpoint.pth.tar'):
    os.makedirs(path, exist_ok=True)
    torch.save(state, '%s/%s' % (path, filename))

    if is_best:
        shutil.copyfile(filename, '%s/%s' % (path, 'best.pth.tar'))



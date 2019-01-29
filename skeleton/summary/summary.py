# -*- coding: utf-8 -*-
from __future__ import absolute_import
import logging

import torch
from .writer import Buffer


LOGGER = logging.getLogger(__name__)


def log(type_, tag, name, tensor, global_step):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.to(device=torch.device('cpu')).detach().numpy()
    Buffer.get(tag)[type_][name] = (tensor, global_step)


def scalar(tag, name, tensor, global_step=None):
    log('scalar', tag, name, tensor, global_step)


def image(tag, name, tensor, global_step=None):
    log('image', tag, name, tensor, global_step)


def text(tag, name, tensor, global_step=None):
    log('text', tag, name, tensor, global_step)


def histogram(tag, name, tensor, global_step=None):
    log('histogram', tag, name, tensor, global_step)


def embedding(tag, name, tensor, global_step=None):
    log('embedding', tag, name, tensor, global_step)

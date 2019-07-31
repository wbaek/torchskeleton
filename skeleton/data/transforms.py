# -*- coding: utf-8 -*-
import os
import logging
import shutil
import hashlib
import collections

import numpy as np
import torch
from PIL import Image


LOGGER = logging.getLogger(__name__)


class Pad:
    def __init__(self, border, mode='reflect'):
        self.border = border
        self.mode = mode

    def __call__(self, image):
        return np.pad(image, [(self.border, self.border), (self.border, self.border), (0, 0)], mode=self.mode)


class Cutout:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, image):
        h, w = image.size(1), image.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.choice(range(h))
        x = np.random.choice(range(w))

        y1 = np.clip(y - self.height // 2, 0, h)
        y2 = np.clip(y + self.height // 2, 0, h)
        x1 = np.clip(x - self.width // 2, 0, w)
        x2 = np.clip(x + self.width // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(image)
        image *= mask
        return image

    def __repr__(self):
        return self.__class__.__name__ + '(height={0}, width={1})'.format(self.height, self.width)


class TensorRandomHorizontalFlip:
    def __call__(self, tensor):
        choice = np.random.choice([True, False])
        return torch.flip(tensor, dims=[-1]) if choice else tensor


class TensorRandomCrop:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, tensor):
        C, H, W = tensor.shape
        h = np.random.choice(range(H + 1 - self.height))
        w = np.random.choice(range(W + 1 - self.width))
        return tensor[:, h:h+self.height, w:w+self.width]


class ImageWriter:
    def __init__(self, root, delete_folder_exists=True):
        self.root = root

        if delete_folder_exists and os.path.exists(self.root):
            shutil.rmtree(self.root)
        os.makedirs(self.root, exist_ok=True)

    def __call__(self, image):
        filename = hashlib.md5(image.tobytes()).hexdigest()
        filepath = os.path.join(self.root,  filename + '.jpg')
        with open(filepath, 'wb') as f:
            image.save(f, format='jpeg')
        return image

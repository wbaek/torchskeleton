# -*- coding: utf-8 -*-
import logging

import numpy as np
import torch


LOGGER = logging.getLogger(__name__)


class Cutout:
    def __init__(self, length):
        self.length = length

    def __call__(self, image):
        h, w = image.size(1), image.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(image)
        image *= mask
        return image

    def __repr__(self):
        return self.__class__.__name__ + '(length={0})'.format(self.length)

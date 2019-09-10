# -*- coding: utf-8 -*-
# pylint: disable=arguments-differ
from __future__ import absolute_import
import logging
from functools import wraps
from collections import OrderedDict

import torch
import numpy as np

LOGGER = logging.getLogger(__name__)


class ToDevice(torch.nn.Module):
    def __init__(self):
        super(ToDevice, self).__init__()
        self.register_buffer('buf', torch.zeros(1, dtype=torch.float32))

    def forward(self, *xs):
        if len(xs) == 1 and isinstance(xs[0], (tuple, list)):
            xs = xs[0]

        device = self.buf.device
        out = []
        for x in xs:
            if x is not None and x.device != device:
                out.append(x.to(device=device))
            else:
                out.append(x)
        return out[0] if len(xs) == 1 else tuple(out)


class CopyChannels(torch.nn.Module):
    def __init__(self, multiple=3, dim=1):
        super(CopyChannels, self).__init__()
        self.multiple = multiple
        self.dim = dim

    def forward(self, x):
        return torch.cat([x for _ in range(self.multiple)], dim=self.dim)


class Normalize(torch.nn.Module):
    def __init__(self, mean, std, inplace=False):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.tensor([mean], dtype=torch.float32)[None, :, None, None])
        self.register_buffer('std', torch.tensor([std], dtype=torch.float32)[None, :, None, None])
        self.inplace = inplace

    def forward(self, x):
        if not self.inplace:
            x = x.clone()

        x.sub_(self.mean).div_(self.std)
        return x


class Permute(torch.nn.Module):
    def __init__(self, *dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class Cutout(torch.nn.Module):
    def __init__(self, ratio=0.0):
        super(Cutout, self).__init__()
        self.ratio = ratio

    def forward(self, tensor):
        batch, channel, height, width = tensor.shape
        w = int(width * self.ratio)
        h = int(height * self.ratio)

        if self.training and w > 0 and h > 0:
            x = np.random.randint(width, size=(batch,))
            y = np.random.randint(height, size=(batch,))

            x1s = np.clip(x - w // 2, 0, width)
            x2s = np.clip(x + w // 2, 0, width)
            y1s = np.clip(y - h // 2, 0, height)
            y2s = np.clip(y + h // 2, 0, height)

            mask = torch.ones_like(tensor)
            for idx, (x1, x2, y1, y2) in enumerate(zip(x1s, x2s, y1s, y2s)):
                mask[idx, :, y1:y2, x1:x2] = 0.

            tensor = tensor * mask
        return tensor


class Mul(torch.nn.Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight

    def forward(self, x):
        return x * self.weight


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def decorator_tuple_to_args(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        args = list(args)
        if len(args) == 2 and isinstance(args[1], (tuple, list)):
            args[1:] = list(args[1])
        return func(*args, **kwargs)
    return wrapper


class Concat(torch.nn.Module):
    def __init__(self, dim=1):
        super(Concat, self).__init__()
        self.dim = dim

    @decorator_tuple_to_args
    def forward(self, *xs):
        return torch.cat(xs, dim=self.dim)


class MergeSum(torch.nn.Module):
    @decorator_tuple_to_args
    def forward(self, *xs):
        return torch.sum(torch.stack(xs), dim=0)


class MergeProd(torch.nn.Module):
    @decorator_tuple_to_args
    def forward(self, *xs):
        return xs[0] * xs[1]


class Choice(torch.nn.Module):
    def __init__(self, idx=0):
        super(Choice, self).__init__()
        self.idx = idx

    @decorator_tuple_to_args
    def forward(self, *xs):
        return xs[self.idx]


class Toggle(torch.nn.Module):
    def __init__(self, module):
        super(Toggle, self).__init__()
        self.module = module
        self.on = True

    def forward(self, x):
        return self.module(x) if self.on else x


class Split(torch.nn.Module):
    def __init__(self, *modules):
        super(Split, self).__init__()
        if len(modules) == 1 and isinstance(modules[0], OrderedDict):
            for key, module in modules[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(modules):
                self.add_module(str(idx), module)

    def forward(self, x):
        return tuple([m(x) for m in self._modules.values()])


class DropPath(torch.nn.Module):
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self._half = False

    def forward(self, x):
        if self.training and self.drop_prob > 0.:
            # shape = list(x.shape[:2]) + [1 for _ in x.shape[2:]]
            shape = list(x.shape[:1]) + [1 for _ in x.shape[1:]]
            keep_prob = 1. - self.drop_prob
            mask = torch.cuda.FloatTensor(*shape).bernoulli_(keep_prob)
            if self._half:
                mask = mask.half()
            x.div_(keep_prob)
            x.mul_(mask)
        return x

    def half(self):
        self._half = True

    def float(self):
        self._half = False


class DelayedPass(torch.nn.Module):
    def __init__(self):
        super(DelayedPass, self).__init__()
        self.register_buffer('keep', None)

    def forward(self, x):
        rv = self.keep  # pylint: disable=access-member-before-definition
        self.keep = x
        return rv


class Reader(torch.nn.Module):
    def __init__(self, x=None):
        super(Reader, self).__init__()
        self.x = x

    def forward(self, x):  # pylint: disable=unused-argument
        return self.x


class KeepByPass(torch.nn.Module):
    def __init__(self):
        super(KeepByPass, self).__init__()
        self._reader = Reader()
        self.info = {}

    @property
    def x(self):
        return self._reader.x

    def forward(self, x):
        self._reader.x = x
        return x

    def reader(self):
        return self._reader


class StrideConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=1, groups=1):
        super(StrideConv2d, self).__init__()
        self.op1 = torch.nn.Conv2d(in_channels, out_channels // 4, kernel_size=kernel_size, stride=2, padding=padding, groups=groups)
        self.op2 = torch.nn.Conv2d(in_channels, out_channels // 4, kernel_size=kernel_size, stride=2, padding=padding, groups=groups)
        self.op3 = torch.nn.Conv2d(in_channels, out_channels // 4, kernel_size=kernel_size, stride=2, padding=padding, groups=groups)
        self.op4 = torch.nn.Conv2d(in_channels, out_channels // 4, kernel_size=kernel_size, stride=2, padding=padding, groups=groups)

    def forward(self, x):
        # y = self.op(x)
        y = x
        y1 = y[:, :, :, :]
        y2 = y[:, :, 1:, :]
        y3 = y[:, :, :, 1:]
        y4 = y[:, :, 1:, 1:]
        return torch.cat([self.op1(y1), self.op2(y2), self.op3(y3), self.op4(y4)], dim=1)


class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SepConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, affine=True, track_running_stats=True):
        super(SepConv, self).__init__()
        self.op = torch.nn.Sequential(
            torch.nn.ReLU6(inplace=True),
            torch.nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False),
            torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(in_channels, affine=affine, track_running_stats=track_running_stats),
            torch.nn.ReLU6(inplace=True),
            torch.nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=padding, groups=in_channels, bias=False),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(out_channels, affine=affine, track_running_stats=track_running_stats),
        )

    def forward(self, x):
        return self.op(x)


class Identity2d(torch.nn.Module):
    def __init__(self, stride=1):
        super(Identity2d, self).__init__()
        self.stride = stride

    def forward(self, x):
        return x if self.stride == 1 else x[:, :, ::self.stride, ::self.stride]


class Skip2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, affine=True, track_running_stats=True):
        super(Skip2d, self).__init__()
        self.in_channels, self.out_channels, self.stride = in_channels, out_channels, stride
        if in_channels != out_channels:
            self.op = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, groups=1, bias=False),
                torch.nn.BatchNorm2d(out_channels, affine=affine, track_running_stats=track_running_stats),
            )
        else:
            self.op = Identity2d(stride=stride)

    def forward(self, x):
        return self.op(x)

    def __repr__(self):
        return 'Skip2d(%d, %d, stride=%s)' % (self.in_channels, self.out_channels, self.stride)


class MBConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, expand_ratio=1, affine=True,
                 track_running_stats=True, se_ratio=None, activation=torch.nn.ReLU6(inplace=True)):
        super(MBConv, self).__init__()
        self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.sxpand_ratio = in_channels, out_channels, kernel_size, stride, padding, expand_ratio
        inter_channels = int(in_channels * expand_ratio)
        if expand_ratio != 1:
            expand_block = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
                torch.nn.BatchNorm2d(inter_channels, affine=affine, track_running_stats=track_running_stats),
                activation,
            )
        else:
            expand_block = torch.nn.Identity()

        if se_ratio is not None and 0 < se_ratio <= 1:
            se_channels = max(1, int(inter_channels * se_ratio))
            se_block = torch.nn.Sequential(
                Split(
                    torch.nn.Sequential(
                        torch.nn.AdaptiveAvgPool2d((1, 1)),
                        torch.nn.Conv2d(inter_channels, se_channels, kernel_size=1),
                        activation,
                        torch.nn.Conv2d(se_channels, inter_channels, kernel_size=1),
                        torch.nn.Sigmoid()
                    ),
                    torch.nn.Identity()
                ),
                MergeProd()
            )
        else:
            se_block = torch.nn.Identity()

        self.op = torch.nn.Sequential(
            expand_block,

            torch.nn.Conv2d(inter_channels, inter_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=inter_channels, bias=False),
            torch.nn.BatchNorm2d(inter_channels, affine=affine, track_running_stats=track_running_stats),
            activation,

            se_block,
            torch.nn.Conv2d(inter_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(out_channels, affine=affine, track_running_stats=track_running_stats),
        )

    def forward(self, x):
        out = self.op(x)
        if x.shape == out.shape:
            out = out + x
        return out

    def __repr__(self):
        return 'MBConv(%d, %d, kernel_size=%s, stride=%s, padding=%s, expand_ratio=%s)' % (self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.sxpand_ratio)


class SEBlock(torch.nn.Module):
    def __init__(self, in_channels, reduction=16, activation=torch.nn.ReLU(inplace=True)):
        super(SEBlock, self).__init__()

        inter_channels = max(1, in_channels // reduction)
        self.op = torch.nn.Sequential(
            Split(
                torch.nn.Sequential(
                    torch.nn.AdaptiveAvgPool2d((1, 1)),
                    torch.nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
                    activation,
                    torch.nn.Conv2d(inter_channels, in_channels, kernel_size=1, bias=False),
                    torch.nn.Sigmoid()
                ),
                torch.nn.Identity()
            ),
            MergeProd()
        )

    def forward(self, x):
        return self.op(x)

# -*- coding: utf-8 -*-
# pylint: disable=arguments-differ, abstract-method
from __future__ import absolute_import
import logging

import torch


LOGGER = logging.getLogger(__name__)


class Operations:
    @staticmethod
    def names():
        return ['none|zero', 'skip|identity', 'pool_(avg|max)_%d', 'conv[_dec|_sep_|_dil_%d]_%d']

    @staticmethod
    def create(name, channels, stride=1, affine=True):
        if name in ['none', 'zero']:
            return Zero(stride)
        if name in ['skip', 'identity']:
            return Identity() if stride == 1 else FactorizedReduce(channels, channels, affine=affine)

        splited = name.split('_')
        if 'pool' in splited:
            kernel = int(splited[-1])
            padding = int((kernel - (kernel % 2)) // 2)
            if 'avg' in splited:
                return torch.nn.AvgPool2d(kernel, stride=stride, padding=padding, count_include_pad=False)
            if 'max' in splited:
                return torch.nn.MaxPool2d(kernel, stride=stride, padding=padding)
        if 'conv' in splited:
            kernel = int(splited[-1])
            padding = int((kernel - (kernel % 2)) // 2)
            dilation = 1
            if 'dil' in splited:
                dilation = int(splited[-2])
                padding = dilation * padding

            if 'sep' in splited:
                return ConvSeparable(channels, channels, kernel, stride, padding, dilation, affine=affine)
            if 'dil' in splited:
                return ConvDilation(channels, channels, kernel, stride, padding, dilation, affine=affine)
            if 'dec' in splited:
                return ConvDecomposition(channels, channels, kernel, stride, padding, dilation, affine=affine)

            return ReLUConvBN(channels, channels, kernel, stride, padding, dilation, affine=affine)

        raise Exception('Ops.create not exists operation %s' % name)


class ReLUConvBN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = torch.nn.Sequential(
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                            dilation=dilation, bias=False),
            torch.nn.BatchNorm2d(out_channels, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class ConvDilation(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, affine=True):
        super(ConvDilation, self).__init__()
        self.op = torch.nn.Sequential(
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                            dilation=dilation, groups=in_channels, bias=False),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(out_channels, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class ConvSeparable(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, affine=True):
        super(ConvSeparable, self).__init__()
        self.op = torch.nn.Sequential(
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                            dilation=dilation, groups=in_channels, bias=False),
            torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(in_channels, affine=affine),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=padding,
                            dilation=dilation, groups=in_channels, bias=False),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(out_channels, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class ConvDecomposition(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, affine=True):
        super(ConvDecomposition, self).__init__()
        self.op = torch.nn.Sequential(
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels, in_channels, (1, kernel_size), stride=(1, stride), padding=(0, padding),
                            dilation=(1, dilation), bias=False),
            torch.nn.Conv2d(in_channels, out_channels, (kernel_size, 1), stride=(stride, 1), padding=(padding, 0),
                            dilation=(dilation, 1), bias=False),
            torch.nn.BatchNorm2d(out_channels, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class Identity(torch.nn.Module):
    def forward(self, x):
        return x


class Zero(torch.nn.Module):
    def __init__(self, stride=1):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(torch.nn.Module):
    def __init__(self, in_channels, out_channels, affine=True):
        super(FactorizedReduce, self).__init__()
        assert out_channels % 2 == 0
        self.relu = torch.nn.ReLU(inplace=False)
        self.conv_1 = torch.nn.Conv2d(in_channels, out_channels // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = torch.nn.Conv2d(in_channels, out_channels // 2, 1, stride=2, padding=0, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out

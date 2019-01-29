# -*- coding: utf-8 -*-
# pylint: disable=wildcard-import, unused-wildcard-import
import pytest

from torch import nn

from skeleton.darts.operations import *


def test_create():
    assert isinstance(Operations.create('none', 10), Zero)
    assert isinstance(Operations.create('zero', 10), Zero)
    assert isinstance(Operations.create('skip', 10), Identity)
    assert isinstance(Operations.create('identity', 10), Identity)

    assert isinstance(Operations.create('pool_avg_3', 10), nn.AvgPool2d)
    assert isinstance(Operations.create('pool_max_3', 10), nn.MaxPool2d)

    assert isinstance(Operations.create('conv_3', 10), ReLUConvBN)
    assert isinstance(Operations.create('conv_sep_3', 10), ConvSeparable)
    assert isinstance(Operations.create('conv_dil_2_3', 10), ConvDilation)
    assert isinstance(Operations.create('conv_dec_3', 10), ConvDecomposition)

    with pytest.raises(Exception):
        Operations.create('not_exist_op', 10)

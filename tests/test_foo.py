# -*- coding: utf-8 -*-
from skeleton.foo import Bar


def test_bar():
    assert Bar().baz() == 'Bar'

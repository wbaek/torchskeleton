# -*- coding: utf-8 -*-


class Bar:
    def __init__(self):
        self.name = self.__class__.__name__

    def baz(self):
        return self.name

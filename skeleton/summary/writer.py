# -*- coding: utf-8 -*-
from __future__ import absolute_import
import logging
import collections

from tensorboardX import SummaryWriter


LOGGER = logging.getLogger(__name__)


class Buffer:
    _instance = None

    @staticmethod
    def get(tag=None):
        if Buffer._instance is None:
            Buffer()
        if tag is None:
            return Buffer._instance
        if tag not in Buffer._instance.buffers:
            Buffer._instance.buffers[tag] = collections.defaultdict(dict)
        return Buffer._instance.buffers[tag]

    @staticmethod
    def close():
        Buffer._instance = None

    def clear(self):
        self.buffers = {tag: collections.defaultdict(dict) for tag in self.buffers.keys()}
        return self

    def __init__(self):
        if Buffer._instance is not None:
            raise Exception('This class is a singleton!')

        self.buffers = {}
        self.clear()
        Buffer._instance = self


Buffer()


class FileWriter():
    def __init__(self, logdir, tags=('train', 'valid', 'test'), comment=''):
        self.tags = tags
        self.writers = {tag: SummaryWriter(logdir + '/' + tag, comment=comment) for tag in self.tags}

    def close(self):
        _ = [w.close() for _, w in self.writers.items()]

    def add_graph(self, tag, model, input_tensor):
        self.writers[tag].add_graph(model, input_tensor)
        return self

    def write(self, global_step):
        for tag in self.tags:
            writer = self.writers[tag]
            for type_, variables in Buffer.get(tag).items():
                if type_ == 'scalar':
                    fn = writer.add_scalar
                elif type_ == 'image':
                    fn = writer.add_image
                elif type_ == 'text':
                    fn = writer.add_text
                elif type_ == 'histogram':
                    fn = writer.add_histogram
                elif type_ == 'embedding':
                    fn = writer.add_embedding
                else:
                    raise LookupError('[%s] not support type at %s' % (self.__class__.__name__, type_))

                for name, (tensor, local_global_step) in variables.items():
                    local_global_step = local_global_step if local_global_step is not None else global_step
                    fn(name, tensor, local_global_step)
        Buffer.get().clear()
        return self

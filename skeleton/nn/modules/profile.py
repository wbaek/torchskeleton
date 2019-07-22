# -*- coding: utf-8 -*-
# pylint: disable=arguments-differ, abstract-method
from __future__ import absolute_import
import logging

import numpy as np
import torch
from torch.nn.modules.conv import _ConvNd


LOGGER = logging.getLogger(__name__)


class Profiler:
    def __init__(self, module):
        self.module = module

    def params(self, name_filter=lambda name: True):
        return np.sum(params.numel() for name, params in self.module.named_parameters() if name_filter(name))

    def flops(self, *inputs, name_filter=lambda name: 'skeleton' not in name and 'loss' not in name):
        operation_flops = []

        def get_hook(name):
            def counting(module, inp, outp):
                class_name = module.__class__.__name__

                fn = None
                module_type = type(module)
                if not name_filter(str(module_type)):
                    pass
                elif isinstance(module, _ConvNd):
                    fn = count_conv_flops
                elif isinstance(module, torch.nn.Linear):
                    fn = count_linear_flops
                else:
                    pass
                    # LOGGER.warning('Not implemented for %s', module_type)

                flops = fn(module, inp, outp) if fn is not None else 0
                data = {
                    'name': name,
                    'class_name': class_name,
                    'flops': flops,
                }
                operation_flops.append(data)
            return counting

        handles = []
        for name, module in self.module.named_modules():
            if len(list(module.children())) > 0:  # pylint: disable=len-as-condition
                continue
            handle = module.register_forward_hook(get_hook(name))
            handles.append(handle)

        with torch.no_grad():
            _ = self.module(*inputs)

        # remove hook
        _ = [h.remove() for h in handles]

        return np.sum([data['flops'] for data in operation_flops if name_filter(data['name'])])


# base code from https://github.com/JaminFong/DenseNAS/blob/master/tools/multadds_count.py
def count_conv_flops(conv_module, input, output):
    input = input[0]

    batch_size = input.shape[0]
    output_height, output_width = output.shape[2:]

    kernel_height, kernel_width = conv_module.kernel_size
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels

    conv_per_position_flops = (kernel_height * kernel_width * in_channels * out_channels) / conv_module.groups

    active_elements_count = batch_size * output_height * output_width

    if hasattr(conv_module, '__mask__') and conv_module.__mask__ is not None:
        # (b, 1, h, w)
        flops_mask = conv_module.__mask__.expand(batch_size, 1, output_height, output_width)
        active_elements_count = flops_mask.sum()

    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0
    if conv_module.bias is not None:
        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops
    return int(overall_flops)


def count_linear_flops(linear_module, input, output):
    input = input[0]
    batch_size = input.shape[0]
    overall_flops = linear_module.in_features * linear_module.out_features * batch_size

    bias_flops = 0
    # if conv_module.bias is not None:
    #     bias_flops = out_channels * active_elements_count

    overall_flops = overall_flops + bias_flops
    return int(overall_flops)

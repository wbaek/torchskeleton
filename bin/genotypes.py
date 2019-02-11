# -*- coding: utf-8 -*-
from collections import OrderedDict

''' original genotype
# from https://github.com/quark0/darts/blob/f276dd346a09ae3160f8e3aca5c7b193fda1da37/cnn/genotypes.py#L75
Genotype(
    normal=[
        ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 1), ('skip_connect', 0),
        ('skip_connect', 0), ('dil_conv_3x3', 2)],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ('max_pool_3x3', 0), ('max_pool_3x3', 1),
        ('skip_connect', 2), ('max_pool_3x3', 1),
        ('max_pool_3x3', 0), ('skip_connect', 2),
        ('skip_connect', 2), ('max_pool_3x3', 1)],
    reduce_concat=[2, 3, 4, 5]
)
'''  # pylint: disable=pointless-string-statement

ORIGINAL_DARTS = OrderedDict([
    ('normal', {
        'path': [
            {'to': 2, 'from': 0, 'name': 'conv_sep_3'},
            {'to': 2, 'from': 1, 'name': 'conv_sep_3'},
            {'to': 3, 'from': 0, 'name': 'conv_sep_3'},
            {'to': 3, 'from': 1, 'name': 'conv_sep_3'},
            {'to': 4, 'from': 0, 'name': 'skip'},
            {'to': 4, 'from': 1, 'name': 'conv_sep_3'},
            {'to': 5, 'from': 0, 'name': 'skip'},
            {'to': 5, 'from': 2, 'name': 'conv_dil_2_3'},
        ],
        'node': [2, 3, 4, 5]
    }),
    ('reduce', {
        'path': [
            {'to': 2, 'from': 0, 'name': 'pool_max_3'},
            {'to': 2, 'from': 1, 'name': 'pool_max_3'},
            {'to': 3, 'from': 1, 'name': 'pool_max_3'},
            {'to': 3, 'from': 2, 'name': 'skip'},
            {'to': 4, 'from': 0, 'name': 'pool_max_3'},
            {'to': 4, 'from': 2, 'name': 'skip'},
            {'to': 5, 'from': 1, 'name': 'pool_max_3'},
            {'to': 5, 'from': 2, 'name': 'skip'},
        ],
        'node': [2, 3, 4, 5]
    })
])

MAML_FROM_CIFAR10 = OrderedDict([
    ('normal', {
        'path': [
            {'to': 2, 'from': 0, 'name': 'conv_sep_3'},
            {'to': 2, 'from': 1, 'name': 'conv_sep_5'},
            {'to': 3, 'from': 1, 'name': 'conv_sep_5'},
            {'to': 3, 'from': 2, 'name': 'conv_sep_5'},
            {'to': 4, 'from': 2, 'name': 'conv_sep_5'},
            {'to': 4, 'from': 3, 'name': 'conv_sep_5'},
            {'to': 5, 'from': 3, 'name': 'conv_sep_5'},
            {'to': 5, 'from': 4, 'name': 'conv_sep_5'},
        ],
        'node': [2, 3, 4, 5]
    }),
    ('reduce', {
        'path': [
            {'to': 2, 'from': 0, 'name': 'conv_sep_5'},
            {'to': 2, 'from': 1, 'name': 'conv_sep_5'},
            {'to': 3, 'from': 1, 'name': 'conv_sep_5'},
            {'to': 3, 'from': 2, 'name': 'conv_sep_5'},
            {'to': 4, 'from': 2, 'name': 'conv_sep_5'},
            {'to': 4, 'from': 3, 'name': 'conv_sep_5'},
            {'to': 5, 'from': 3, 'name': 'conv_sep_5'},
            {'to': 5, 'from': 4, 'name': 'conv_sep_5'},
        ],
        'node': [2, 3, 4, 5]
    })
])

MAML_FROM_CIFAR100 = OrderedDict([
    ('normal', {
        'path': [
            {'to': 2, 'from': 0, 'name': 'conv_sep_5'},
            {'to': 2, 'from': 1, 'name': 'conv_sep_3'},
            {'to': 3, 'from': 1, 'name': 'skip'},
            {'to': 3, 'from': 2, 'name': 'conv_sep_5'},
            {'to': 4, 'from': 2, 'name': 'conv_sep_5'},
            {'to': 4, 'from': 3, 'name': 'conv_sep_5'},
            {'to': 5, 'from': 3, 'name': 'conv_sep_5'},
            {'to': 5, 'from': 4, 'name': 'conv_sep_5'},
        ],
        'node': [2, 3, 4, 5]
    }),
    ('reduce', {
        'path': [
            {'to': 2, 'from': 0, 'name': 'conv_sep_5'},
            {'to': 2, 'from': 1, 'name': 'conv_sep_5'},
            {'to': 3, 'from': 1, 'name': 'conv_sep_5'},
            {'to': 3, 'from': 2, 'name': 'conv_sep_5'},
            {'to': 4, 'from': 2, 'name': 'conv_sep_5'},
            {'to': 4, 'from': 3, 'name': 'conv_sep_5'},
            {'to': 5, 'from': 3, 'name': 'conv_sep_5'},
            {'to': 5, 'from': 4, 'name': 'conv_sep_5'},
        ],
        'node': [2, 3, 4, 5]
    })
])

MAML_STOPGRAD_FROM_CIFAR10 = OrderedDict([
    ('normal', {
        'path': [
            {'to': 2, 'from': 0, 'name': 'conv_sep_3'},
            {'to': 2, 'from': 1, 'name': 'conv_sep_5'},
            {'to': 3, 'from': 1, 'name': 'pool_max_3'},
            {'to': 3, 'from': 2, 'name': 'conv_sep_5'},
            {'to': 4, 'from': 2, 'name': 'conv_sep_5'},
            {'to': 4, 'from': 3, 'name': 'conv_sep_5'},
            {'to': 5, 'from': 3, 'name': 'conv_sep_5'},
            {'to': 5, 'from': 4, 'name': 'conv_sep_5'},
        ],
        'node': [2, 3, 4, 5]
    }),
    ('reduce', {
        'path': [
            {'to': 2, 'from': 0, 'name': 'conv_sep_5'},
            {'to': 2, 'from': 1, 'name': 'conv_sep_5'},
            {'to': 3, 'from': 1, 'name': 'conv_sep_5'},
            {'to': 3, 'from': 2, 'name': 'conv_sep_5'},
            {'to': 4, 'from': 2, 'name': 'conv_sep_5'},
            {'to': 4, 'from': 3, 'name': 'conv_sep_5'},
            {'to': 5, 'from': 3, 'name': 'conv_sep_5'},
            {'to': 5, 'from': 4, 'name': 'conv_sep_5'},
        ],
        'node': [2, 3, 4, 5]
    })
])

MAML_STOPGRAD_FROM_CIFAR100 = OrderedDict([
    ('normal', {
        'path': [
            {'to': 2, 'from': 0, 'name': 'conv_sep_5'},
            {'to': 2, 'from': 1, 'name': 'conv_sep_5'},
            {'to': 3, 'from': 1, 'name': 'pool_max_3'},
            {'to': 3, 'from': 2, 'name': 'conv_sep_5'},
            {'to': 4, 'from': 2, 'name': 'conv_sep_5'},
            {'to': 4, 'from': 3, 'name': 'conv_sep_5'},
            {'to': 5, 'from': 3, 'name': 'conv_sep_5'},
            {'to': 5, 'from': 4, 'name': 'conv_sep_5'},
        ],
        'node': [2, 3, 4, 5]
    }),
    ('reduce', {
        'path': [
            {'to': 2, 'from': 0, 'name': 'conv_sep_5'},
            {'to': 2, 'from': 1, 'name': 'conv_sep_5'},
            {'to': 3, 'from': 1, 'name': 'conv_sep_5'},
            {'to': 3, 'from': 2, 'name': 'conv_sep_5'},
            {'to': 4, 'from': 2, 'name': 'conv_sep_5'},
            {'to': 4, 'from': 3, 'name': 'conv_sep_5'},
            {'to': 5, 'from': 3, 'name': 'conv_sep_5'},
            {'to': 5, 'from': 4, 'name': 'conv_sep_5'},
        ],
        'node': [2, 3, 4, 5]
    })
])

MAML_CUTOUT_FROM_CIFAR100 = OrderedDict([
    ('normal', {
        'path': [
            {'to': 2, 'from': 0, 'name': 'conv_sep_5'},
            {'to': 2, 'from': 1, 'name': 'conv_sep_5'},
            {'to': 3, 'from': 1, 'name': 'conv_sep_5'},
            {'to': 3, 'from': 2, 'name': 'conv_sep_5'},
            {'to': 4, 'from': 2, 'name': 'conv_sep_5'},
            {'to': 4, 'from': 3, 'name': 'conv_sep_5'},
            {'to': 5, 'from': 3, 'name': 'conv_sep_5'},
            {'to': 5, 'from': 4, 'name': 'conv_sep_5'},
        ],
        'node': [2, 3, 4, 5]
    }),
    ('reduce', {
        'path': [
            {'to': 2, 'from': 0, 'name': 'conv_sep_5'},
            {'to': 2, 'from': 1, 'name': 'conv_sep_5'},
            {'to': 3, 'from': 1, 'name': 'conv_sep_5'},
            {'to': 3, 'from': 2, 'name': 'conv_sep_5'},
            {'to': 4, 'from': 2, 'name': 'conv_sep_5'},
            {'to': 4, 'from': 3, 'name': 'conv_sep_5'},
            {'to': 5, 'from': 3, 'name': 'conv_sep_5'},
            {'to': 5, 'from': 4, 'name': 'conv_dil_2_5'},
        ],
        'node': [2, 3, 4, 5]
    })
])

MAML_REPTILE_FROM_CIFAR100 = OrderedDict([
    ('normal', {
        'path': [
            {'to': 2, 'from': 0, 'name': 'conv_sep_3'},
            {'to': 2, 'from': 1, 'name': 'conv_sep_5'},
            {'to': 3, 'from': 1, 'name': 'pool_max_3'},
            {'to': 3, 'from': 2, 'name': 'conv_sep_5'},
            {'to': 4, 'from': 1, 'name': 'pool_max_3'},
            {'to': 4, 'from': 2, 'name': 'conv_sep_5'},
            {'to': 5, 'from': 3, 'name': 'pool_max_3'},
            {'to': 5, 'from': 4, 'name': 'conv_sep_5'},
        ],
        'node': [2, 3, 4, 5]
    }),
    ('reduce', {
        'path': [
            {'to': 2, 'from': 0, 'name': 'conv_sep_5'},
            {'to': 2, 'from': 1, 'name': 'conv_sep_5'},
            {'to': 3, 'from': 1, 'name': 'conv_sep_5'},
            {'to': 3, 'from': 2, 'name': 'conv_sep_5'},
            {'to': 4, 'from': 2, 'name': 'conv_sep_5'},
            {'to': 4, 'from': 3, 'name': 'conv_sep_5'},
            {'to': 5, 'from': 3, 'name': 'conv_sep_5'},
            {'to': 5, 'from': 4, 'name': 'conv_sep_5'},
        ],
        'node': [2, 3, 4, 5]
    })
])

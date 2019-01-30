# -*- coding: utf-8 -*-
import os
from setuptools import setup, find_packages

DOC_NAME = 'README.md'

long_description = """
"""

if os.path.isfile(DOC_NAME):
    with open(DOC_NAME) as fp:
        long_description = fp.read()

long_description = long_description or ''

setup(
    long_description=long_description,
    packages=find_packages(),
    # auto generated:
    name='skeleton',
    version='0.1.0',
    description='',
    keywords=['pytorch', 'skeleton'],
    author='clint',
    author_email='clint.b@kakaobrain.com',
    url='https://github.com/wbaek/pytorch_skeleton',
    classifiers=[],
    scripts=[],
    entry_points={},
    zip_safe=False,
    include_package_data=True,
    setup_requires=[],
    data_files=[
        ('', [
            'skeleton/datasets/datas/ILSVRC/classification/imagenet1000_classid_to_text_synsetid.json',
            'skeleton/datasets/datas/ILSVRC/classification/imagenet_2012_validation_synset_labels.txt',
        ])
    ],
    install_requires=[
        'torch',
        'torchvision',
        'skorch',
        'tensorboardX',
        'tqdm',
        'treelib',
        'opencv-python',
    ],
    dependency_links=[
        'https://github.com/wbaek/theconf/tarball/master',
    ],
    tests_require=[],
)

#!/usr/bin/env python
import os
import shutil
import sys
from setuptools import setup, find_packages

VERSION = '0.0.1'

long_description = "an abstraction for logging neural network expirements"

setup_info = dict(
    # Metadata
    name='pytorchart',
    version=VERSION,
    author='Pavel Savine',
    author_email='psavine42@gmail.com',
    url='https://github.com/psavine42/pytorchart',
    description=long_description,
    long_description=long_description,
    license='BSD',

    # Package info
    # dependency_links=['http://github.com/pytorch/tnt/tarball/master#egg=package-1.0'],
    packages=find_packages(exclude=('test', 'imgs')),
    zip_safe=True,
    install_requires=[
        'torch',
        'six',
        'visdom',
        # 'torchnet'
    ]
)

setup(**setup_info)
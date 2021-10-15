#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name='Pytorch-QNN',
    version='1',
    license='GNU v3.0',
    packages=find_packages() + find_packages('core_qnn/'),
    package_dir={'core_qnn': 'core_qnn'},
)

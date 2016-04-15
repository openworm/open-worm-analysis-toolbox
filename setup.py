#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
setup.py for the Python packager

Based on https://github.com/pypa/sampleproject/blob/master/setup.py
"""

# Always prefer setuptools over distutils
from setuptools import setup
# To use a consistent encoding
from codecs import open
from os import path
import os

# Get __version__ without having to call all the extra modules in __init__
# you would get from:
# from open_worm_analysis_toolbox.version import __version__
exec(open('open_worm_analysis_toolbox/version.py').read())

here = path.abspath(path.dirname(__file__))
readme_path = path.join(here, 'README.md')

long_description = 'See https://github.com/openworm/open-worm-analysis-toolbox\n'

# Get the long description from the README file, and add it.
if path.exists(readme_path):
    with open(readme_path, encoding='utf-8') as f:
        long_description += f.read()

print(os.listdir('.'))  # DEBUG

setup(
    name='open_worm_analysis_toolbox',
    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=__version__,
    description='Open Worm Analysis Toolbox',
    long_description=long_description,
    url='https://github.com/openworm/open-worm-analysis-toolbox',
    author=('Yemini, E; Jucikas, T; Schafer, W; Brown, A; Hokanson, J; '
            'Currie, M; Javer, A; OpenWorm'),
    author_email='mcurrie@openworm.org',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    keywords='C. elegans worm tracking',
    packages=['open_worm_analysis_toolbox'],
    install_requires=['atlas', 'nose', 'pandas', 'statsmodels',
                      'h5py', 'seaborn']
    # Actually also requires openCV, numpy, scipy, matplotlib and numpy
    # but I don't want to force pip to install these here since pip is bad
    # at that for those packages.
)

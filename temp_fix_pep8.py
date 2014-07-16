# -*- coding: utf-8 -*-
"""

"""

import glob
import os
from os import path
import autopep8 as ap

# See options at:
# https://pypi.python.org/pypi/autopep8/#usage
#-i : in place editing

options = ap.parse_args(['-i', ''])

try:
    cur_path = path.dirname(path.realpath(__file__))
except NameError:
    # ASSUMES IN ROOT PATH - i.e. that wormpy package is in this folder
    cur_path = os.getcwd()

wormpy_path = path.join(cur_path, 'wormpy')
stats_path  = path.join(wormpy_path,'stats')

print(wormpy_path)

wormpy_files = glob.glob(path.join(wormpy_path, '*.py'))
root_files = glob.glob(path.join(cur_path, '*.py'))
stats_files = glob.glob(path.join(stats_path, '*.py'))


ap.fix_multiple_files(wormpy_files,options=options)
ap.fix_multiple_files(root_files, options=options)
ap.fix_multiple_files(stats_files, options=options)


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
#-r : See https://github.com/hhatto/autopep8/issues/30


#TODO   *************
#Might want to disable comment editing

options = ap.parse_args(['-i', ''])
#options = ap.parse_args(['-i', '-r', ''])


try:
  cur_path = path.dirname(path.realpath(__file__))  
except NameError:
  #ASSUMES IN ROOT PATH - i.e. that wormpy package is in this folder
  cur_path = os.getcwd()
  
wormpy_path = path.join(cur_path,'wormpy')
  
print(wormpy_path)  

wormpy_files = glob.glob(path.join(wormpy_path,'*.py'))
root_files   = glob.glob(path.join(wormpy_path,'*.py'))
 
ap.fix_multiple_files(wormpy_files,options=options) 
  
#all_files = ['jim_test.py']
#
#
#
#
#print glob.glob("./wormpy/*.py")
#
#all_files = [
#    'config.py',
#    'Events.py',
#    'feature_comparisons.py']
#
#
#file_path = path.relpath("jim_test.py")
#wtf = ap.fix_file(file_path,options=options)
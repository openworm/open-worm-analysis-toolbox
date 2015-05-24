# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 22:29:15 2015

#TODO: document what this code does ...

Credit to Christopher R. Wagner at 
http://robotfantastic.org/serializing-python-data-to-json-some-edge-cases.html    
for the following six functions:

isnamedtuple
serialize
restore
data_to_json
json_to_data
nested_equal

"""
import sys, os
import warnings

# We must add .. to the path so that we can perform the 
# import of movement_validation while running this as 
# a top-level script (i.e. with __name__ = '__main__')
sys.path.append('..') 
from movement_validation import user_config

from movement_validation.basic_worm import BasicWorm
from movement_validation import NormalizedWorm


def main():
    warnings.filterwarnings('error')
    
    nw = get_nw()
    
    bw = nw.get_BasicWorm()
    
    nw_calculated = NormalizedWorm.from_BasicWorm_factory(bw)

       
def main1():
    warnings.filterwarnings('error')
    
    base_path = os.path.abspath(user_config.EXAMPLE_DATA_PATH)

    JSON_path = os.path.join(base_path, 'test.JSON')

    b = BasicWorm()
    #b.contour[0] = 100.2
    #b.metadata['vulva'] = 'CCW'
    b.save_to_JSON(JSON_path)

    c = BasicWorm()
    c.load_from_JSON(JSON_path)
    print(c.contour)

    #dat.save_to_JSON(JSON_path)


def get_nw():
    base_path = os.path.abspath(user_config.EXAMPLE_DATA_PATH)
    schafer_file_path = os.path.join(base_path, 
                                     'example_video_norm_worm.mat')
    nw = NormalizedWorm.from_schafer_file_factory(schafer_file_path)


if __name__ == '__main__':
    main()
    
    
    
    
    
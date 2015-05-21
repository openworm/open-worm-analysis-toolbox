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
import numpy as np

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

    schafer_file_path = os.path.join(
        base_path, 'example_video_norm_worm.mat')

    return NormalizedWorm.from_schafer_file_factory(schafer_file_path)


def main2():
    JSON_path0 = 'C:\\Users\\Michael\\Dropbox\\INBOX\\test0.JSON'
    JSON_path = 'C:\\Users\\Michael\\Dropbox\\INBOX\\test.JSON'
    
    data_array = np.arange(10)
    x = data_array    

    # METHOD 1
    serialized_data = json.dumps(data_array.tolist())
    with open(JSON_path0, 'w') as outfile:
        outfile.write(serialized_data)

    with open(JSON_path0, 'r') as infile:
        y = json.loads(infile.read())
    
    print("x:", str(type(x)), x)
    print("y:", str(type(y)), y)
    print("x==y:", np.array_equal(x,y))

    
    # METHOD 2
    #http://robotfantastic.org/serializing-python-data-to-json-some-edge-cases.html
    serialized_data = json.dumps(data_array.tolist())
    y = json.loads(serialized_data)



    print("x:", str(type(x)), x)
    print("y:", str(type(y)), y)
    print("x==y:", np.array_equal(x,y))
        

    b = BasicWorm()
    b_as_list = list(b.__dict__.items())
    serialized_data = data_to_json(b_as_list)
    print(serialized_data)

    with open(JSON_path, 'w') as outfile:
        outfile.write(serialized_data)

    with open(JSON_path, 'r') as infile:
        y = json.loads(infile.read())


    

    # TODO: make sure you can read it back!!!
    # TODO: then you just need to make the specification and you are done!

    # TODO: somehow replace the equality check with using nested_equal
    #       on the serializations of the two objects?? maybe.

    #print(json.dumps(b))
    #b_encoded = jsonpickle.encode(b, unpicklable=False)
    #print(b_encoded)





if __name__ == '__main__':
    main()
    
    
    
    
    
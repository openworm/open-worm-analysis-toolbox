# -*- coding: utf-8 -*-
"""
Some unit tests of the movement_validation package

"""

import sys

# We must add .. to the path so that we can perform the 
# import of movement_validation while running this as 
# a top-level script (i.e. with __name__ = '__main__')
sys.path.append('..') 
#from movement_validation import NormalizedWorm


def test_simply():
    # Simple test to verify our test harness works
    assert(1 == 1)



def test_empty_nw():
    # Test edge cases when our normalized worm is empty
    pass
    #nw = NormalizedWorm()
    #nw.validate()
    #centred_skeleton = nw.centre()
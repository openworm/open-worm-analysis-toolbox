# -*- coding: utf-8 -*-
"""
Some unit tests of the movement_validation package

"""

import sys


# We must add .. to the path so that we can perform the
# import of movement_validation while running this as
# a top-level script (i.e. with __name__ = '__main__')
sys.path.append('..')
import movement_validation as mv
import examples

def test_simply():
    # Simple test to verify our test harness works
    assert(1 == 1)



def test_empty_nw():
    # Test edge cases when our normalized worm is empty
    nw = mv.NormalizedWorm()
    nw.validate()
    #centred_skeleton = nw.centre()

def test_example_scripts():
    """
    Generates a test for each example enumerated in examples_list.
    Tests pass if the exitcode for the example is 0.
    Scripts that do not error exit with 0 by default.
    """

    EXAMPLES_TO_RUN = ['validate_features']    
    
    for example in EXAMPLES_TO_RUN:
        method_to_call = getattr(examples,example)
        method_to_call()



# Unit tests for utils
def test_round_to_odd():
    round_to_odd = mv.utils.round_to_odd
    
    assert(round_to_odd(3) == 3)
    assert(round_to_odd(3.5) == 3)
    assert(round_to_odd(4) in (3,5))
    assert(round_to_odd(-12) in (-11,-13))    


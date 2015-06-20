# -*- coding: utf-8 -*-
"""
Some unit tests of the movement_validation package

"""

import sys, subprocess, os


# We must add .. to the path so that we can perform the
# import of movement_validation while running this as
# a top-level script (i.e. with __name__ = '__main__')
sys.path.append('..')
from movement_validation import NormalizedWorm


def test_simply():
    # Simple test to verify our test harness works
    assert(1 == 1)



def test_empty_nw():
    # Test edge cases when our normalized worm is empty
    nw = NormalizedWorm()
    nw.validate()
    #centred_skeleton = nw.centre()

def test_example_scripts():
    """
    Generates a test for each example enumerated in examples_list.
    Tests pass if the exitcode for the example is 0.
    Scripts that do not error exit with 0 by default.
    """
    # Move to "examples" directory
    os.chdir('examples')
    # Examples that you want to test must be enumerated in examples_list
    # Limit this to examples that exit with a status code. Example scripts
    # must exit, so some may not work (ex: those that produce graphs).
    # sys.exit(*statuscode*) could help in those cases.
    examples_list = ['validate_features.py', 'feature_processing_options_testing.py']
    for example in examples_list:
        yield run_example, example

def run_example(example):
    # Each example generates a test using this function.
    returncode = subprocess.call(['/usr/bin/env', 'python', example], stdout=open(os.devnull, 'wb'))
    assert returncode == 0

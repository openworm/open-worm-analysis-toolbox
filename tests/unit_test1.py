# -*- coding: utf-8 -*-
"""
Some unit tests of the open-worm-analysis-toolbox package

"""
import sys
import os

# We must add .. to the path so that we can perform the
# import of open_worm_analysis_toolbox while running this as
# a top-level script (i.e. with __name__ = '__main__')
sys.path.append('..')
import open_worm_analysis_toolbox as mv
#import scipy as sp


def test_simple():
    # Simple test to verify our test harness works
    assert(1 == 1)


def test_empty_nw():
    # Test edge cases when our normalized worm is empty
    nw = mv.NormalizedWorm()
    nw.validate()
    #centred_skeleton = nw.centre()

# Unit tests for utils


def test_round_to_odd():
    round_to_odd = mv.utils.round_to_odd

    assert(round_to_odd(3) == 3)
    assert(round_to_odd(3.5) == 3)
    assert(round_to_odd(4) in (3, 5))
    assert(round_to_odd(-12) in (-11, -13))


def test_ttest():
    # From http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/
    # scipy.stats.ttest_ind.html
    # TODO
    #rvs1 = sp.stats.norm.rvs(loc=5,scale=10,size=500)
    #rvs2 = sp.stats.norm.rvs(loc=5,scale=10,size=500)
    #p1 = sp.stats.ttest_ind(rvs1, rvs2)
    #p2 = sp.stats.ttest_ind(rvs1, rvs2)

    assert(True)

if __name__ == '__main__':
    print('RUNNING TEST ' + os.path.split(__file__)[1] + ':')

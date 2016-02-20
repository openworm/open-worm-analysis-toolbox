# -*- coding: utf-8 -*-
"""
Package-level configuration settings. The original Schafer Lab code
contained many hardcoded values.  These are tracked in a central
location here.

"""

from __future__ import division

""" DEBUG MODE TO RESTORE OLD SCHAFER LAB ERRORS """

# TODO

""" PRE-FEATURES CONFIGURATION SETTINGS """

# This is the frame rate of the test video.  Generally the FPS should be
# obtained properly from the video itself; this value will not be correct
# for most videos.
DEFAULT_FPS = 25.8398

# Again, generally ventral_mode should be determined and specified by the
# experimenter.  This default value will be wrong for most videos.
DEFAULT_VENTRAL_MODE = 0

N_POINTS_NORMALIZED = 49


""" FEATURES CONFIGURATION SETTINGS """

# Note: for features-level configuration options, see
# features/feature_processing_options.py

EIGENWORM_FILE = 'master_eigen_worms_N2.mat'


""" STATISTICS CONFIGURATION SETTINGS """

# Used in Histogram.h_computeMHists
MAX_NUM_HIST_OBJECTS = 1000

# Used in HistogramManager.h__computeBinInfo
# The maximum # of bins that we'll use. Since the data
# is somewhat random, outliers could really chew up memory. I'd prefer not
# to have some event which all of a sudden tells the computer we need to
# allocate a few hundred gigabytes of data. If this does ever end up a
# problem we'll need a better solution (or really A solution)
MAX_NUMBER_BINS = 10**6

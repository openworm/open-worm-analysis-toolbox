# -*- coding: utf-8 -*-
"""
Package-level configuration settings. The original Schafer Lab code 
contained many hardcoded values.  These are tracked in a central 
location here.

THIS IS BEING PHASED OUT
- to be replaced with movement_validation/features/feature_processing_options.py

Notes
---------------------------------------
Some hardcoded values were not implemented in this open-source translation.
these are shown here as being commented out.

Usage
---------------------------------------
Best practice is to use this via "from movement_validation import config", 
and then reference the configuration settings like this: e.g. "config.FPS" 
rather than doing "from wormpy.config import *" and referencing "FPS"
since the latter approach pollutes the global namespace.
  
"""

from __future__ import division

#__ALL__ = ['FPS', 'N_ECCENTRICITY']

""" DEBUG MODE TO RESTORE OLD SCHAFER LAB ERRORS """


""" FEATURE CONFIGURATION SETTINGS """

# Frames Per Second
# (must be a multiple of both 1/TIP_DIFF and 1/BODY_DIFF)
# JAH to MC; Why?
# This is the frame rate of the test video.
# TODO: This needs to be moved elsewhere, in fact made into a parameter
FPS = 25.8398
# DEBUG: might not need to be here but used in Path code and Locomotion code
DEFAULT_VENTRAL_MODE = 0

""" Locomotion Features """
# used in get_velocity:
TIP_DIFF = 0.25
BODY_DIFF = 0.5




# The following two are used in EventOutputStructure...
# When retrieving the final structure
# this is the name given to the field that contains the
# sum of the input data during the event
DATA_SUM_NAME = 'distance'
# same as above but for BETWEEN events
INTER_DATA_SUM_NAME = 'interDistance'


#======================================================
#                       STATISTICS
#======================================================
# Used in Histogram.h_computeMHists
MAX_NUM_HIST_OBJECTS = 1000

# Used in HistogramManager.h__computeBinInfo
# The maximum # of bins that we'll use. Since the data
# is somewhat random, outliers could really chew up memory. I'd prefer not
# to have some event which all of a sudden tells the computer we need to
# allocate a few hundred gigabytes of data. If this does ever end up a
# problem we'll need a better solution (or really A solution)
MAX_NUMBER_BINS = 10**6



EIGENWORM_FILE = 'master_eigen_worms_N2.mat'


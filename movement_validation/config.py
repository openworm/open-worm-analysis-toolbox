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

MIMIC_OLD_BEHAVIOUR = True

""" FEATURE CONFIGURATION SETTINGS """

# Frames Per Second
# (must be a multiple of both 1/TIP_DIFF and 1/BODY_DIFF)
# JAH to MC; Why?
# This is the frame rate of the test video.
# TODO: This needs to be moved elsewhere, in fact made into a parameter
FPS = 25.8398
# DEBUG: might not need to be here but used in Path code and Locomotion code
DEFAULT_VENTRAL_MODE = 0


""" 
---------------------------------------------
----------    Posture Features     ----------
---------------------------------------------
"""
# posture_features.get_worm_kinks
KINK_LENGTH_THRESHOLD_PCT = 1 / 12  # This the fraction of the worm length
# that a bend must be in order to be counted. The # of worm points
#(this_value*worm_length_in_samples) is rounded to an integer value.
# The threshold value is inclusive.

# posture_features.get_eccentricity_and_orientation
N_ECCENTRICITY = 50  # Grid size for estimating eccentricity, this is the
# max # of points that will fill the wide dimension.
# (scalar) The # of points to place in the long dimension. More points
# gives a more accurate estimate of the ellipse but increases
# the calculation time.


POSTURE_AMPLITURE_AND_WAVELENGTH = {
    'N_POINTS_FFT': 512,
    # NOTE: Unfortunately the distance is in normalized
    # frequency units (indices really), not in real frequency units
    'MIN_DIST_PEAKS': 5,
    'WAVELENGTH_PCT_MAX_CUTOFF': 0.5,  # TODO: describe
    'WAVELENGTH_PCT_CUTOFF': 2}        # TODO: describe

POSTURE_AMPLITURE_AND_WAVELENGTH['HALF_N_FFT'] = \
    POSTURE_AMPLITURE_AND_WAVELENGTH['N_POINTS_FFT'] / 2


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

# Whether to use >= or > as the comparison:
INCLUDE_AT_SPEED_THRESHOLD = True
INCLUDE_AT_DISTANCE_THRESHOLD = True
INCLUDE_AT_FRAMES_THRESHOLD = False
INCLUDE_AT_INTER_FRAMES_THRESHOLD = False


# used in WormPosture
N_EIGENWORMS_USE = 6
EIGENWORM_FILE = 'master_eigen_worms_N2.mat'




# STATISTICS

# Used in Histogram.h_computeMHists
MAX_NUM_HIST_OBJECTS = 1000

# Used in HistogramManager.h__computeBinInfo
# The maximum # of bins that we'll use. Since the data
# is somewhat random, outliers could really chew up memory. I'd prefer not
# to have some event which all of a sudden tells the computer we need to
# allocate a few hundred gigabytes of data. If this does ever end up a
# problem we'll need a better solution (or really A solution)
MAX_NUMBER_BINS = 10**6

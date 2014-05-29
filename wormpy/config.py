# -*- coding: utf-8 -*-
"""
  config.py:
    wormpy module configuration settings
  
  @authors: @JimHokanson, @MichaelCurrie
  
  The original Schafer Lab code contained many hardcoded values.  These are
  tracked in a central location here.
  Some hardcoded values were not implemented in this open-source translation.
  these are shown here as being commented out.
  
  Best practice is to use this via "from wormpy import config", and then 
  reference the configuration settings like this: e.g. "config.FPS" rather
  than doing "from wormpy.config import *" and referencing "FPS"
  since the latter approach pollutes the global namespace.
  
"""

from __future__ import division

#__ALL__ = ['FPS', 'N_ECCENTRICITY']

""" DEBUG MODE TO RESTORE OLD SCHAFER LAB ERRORS """

MIMIC_OLD_BEHAVIOUR = False

""" FEATURE CONFIGURATION SETTINGS """

# Frames Per Second
# (must be a multiple of both 1/TIP_DIFF and 1/BODY_DIFF)
FPS = 20                 
VENTRAL_MODE = 0   # DEBUG: might not need to be here but used in Path code



""" Posture Features """
#posture_features.get_worm_kinks
KINK_LENGTH_THRESHOLD_PCT = 1/12 #This the fraction of the worm length
#that a bend must be in order to be counted. Value is rounded to an 
#integer sample. Threshold is inclusive.


    
N_ECCENTRICITY = 50 # Grid size for estimating eccentricity, this is the
# max # of points that will fill the wide dimension.
# (scalar) The # of points to place in the long dimension. More points
# gives a more accurate estimate of the ellipse but increases
# the calculation time.
 

POSTURE_AMPLITURE_AND_WAVELENGTH = { \
  'N_POINTS_FFT': 512, 
  # NOTE: Unfortunately the distance is in normalized
  # frequency units (indices really), not in real frequency units
  'MIN_DIST_PEAKS': 5, 
  'WAVELENGTH_PCT_MAX_CUTOFF': 0.5,  # TODO: describe
  'WAVELENGTH_PCT_CUTOFF': 2}        # TODO: describe
  
POSTURE_AMPLITURE_AND_WAVELENGTH['HALF_N_FFT'] = \
  POSTURE_AMPLITURE_AND_WAVELENGTH['N_POINTS_FFT']/2

# used in get_velocity:
TIP_DIFF  = 0.25
BODY_DIFF = 0.5


# Used in get_motion_codes:
#-------------------------------
# Interpolate only this length of NaN run; anything longer is
# probably an omega turn.
# If set to "None", interpolate all lengths (i.e. infinity)
MOTION_CODES_LONGEST_NAN_RUN_TO_INTERPOLATE = None
# These are a percentage of the length
SPEED_THRESHOLD_PCT   = 0.05
DISTANCE_THRSHOLD_PCT = 0.05
PAUSE_THRESHOLD_PCT   = 0.025
#   These are times (s)
EVENT_FRAMES_THRESHOLD = 0.5    # Half a second
EVENT_MIN_INTER_FRAMES_THRESHOLD = 0.25
# Used in EventFinder
DATA_SUM_NAME       = 'distance'
INTER_DATA_SUM_NAME = 'interDistance'






# used in WormPosture
N_EIGENWORMS_USE = 6 
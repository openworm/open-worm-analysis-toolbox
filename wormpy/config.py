# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 16:20:47 2013

@author: @MichaelCurrie

The original Schafer Lab code contained many hardcoded values.  These are
tracked in a central location here.
Some hardcoded values were not implemented in this open-source translation.
these are shown here
"""

__ALL__ = ['FPS', 'N_ECCENTRICITY']


""" FEATURE CONFIGURATION SETTINGS """

# Frames Per Second
# (must be a multiple of both 1/TIP_DIFF and 1/BODY_DIFF)
FPS = 20                 

# Grid size for estimating eccentricity, this is the
# max # of points that will fill the wide dimension.
# (scalar) The # of points to place in the long dimension. More points
# gives a more accurate estimate of the ellipse but increases
# the calculation time.
N_ECCENTRICITY = 50     

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
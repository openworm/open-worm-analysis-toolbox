# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 22:36:32 2013

@authors: @JimHokanson, @MichaelCurrie

A translation of Matlab code written by Jim Hokanson,
in the SegwormMatlabClasses GitHub repo.  Original code path:
SegwormMatlabClasses / 
+seg_worm / @feature_calculator / get_features_rewritten.m

NOTE: REQUIRES numpy.version.full_version >= '1.8' 
since numpy.nanmean is only available after that version.
(http://docs.scipy.org/doc/numpy-dev/reference/generated/numpy.nanmean.html)
Alternatively, you can just install nanfunctions.py 
(see instructions in ..//README.md in this repo)

*** For +seg_worm / @feature_calculator / getPostureFeatures.m,
*** here are some renamed variables:

SI = seg_worm.skeleton_indices is expressed here as self.skeleton_partitions
ALL_INDICES = SI.ALL_NORMAL_INDICES is expressed here as 
              self.normal_partitions()
FIELDS = SI.ALL_NORMAL_NAMES is expressed here as 
         self.normal_partitions().keys()
n_fields = length(FIELDS) = len(self.normal_partitions().keys())

"""
import numpy as np
import collections
from wormpy import config
from wormpy import feature_helpers

# TODO: WormFeatures should INHERIT from NormalizedWorm

class WormFeatures:
  """ WormFeatures takes as input a NormalizedWorm instance, and
      during initialization calculates all the features of the worm.
  """
  normalized_worm = None
  
  features = None
  
  morphology = None  # a python dictionary
  locomotion = None
  posture = None
  path = None

  
  def __init__(self, nw):
    self.normalized_worm = nw
    
    self.get_morphology_features()
    self.get_posture_features()    
    self.get_locomotion_features()
    self.get_path_features()
    
      
  def get_morphology_features(self):
    """
    Translation of: SegwormMatlabClasses / 
    +seg_worm / @feature_calculator / getMorphologyFeatures.m
  
    Nature Methods Description
    =======================================================================
   
    Morphology Features 
   
    1. Length. Worm length is computed from the segmented skeleton by
    converting the chain-code pixel length to microns.
    
    2. Widths. Worm width is computed from the segmented skeleton. The
    head, midbody, and tail widths are measured as the mean of the widths
    associated with the skeleton points covering their respective sections.
    These widths are converted to microns.
    
    3. Area. The worm area is computed from the number of pixels within the
    segmented contour. The sum of the pixels is converted to microns2.
   
    4. Area/Length.
   
    5. Midbody Width/Length.
    
    get_morphology_features:
    * Takes normalized_worm, and generates a structure called "morphology"
    
    %Old files that served as a reference ...
    %------------------------------------------------------------
    %morphology_process.m
    %schaferFeatures_process.m
    """
    nw = self.normalized_worm   # just so we have a shorter name to refer to    
    
    self.morphology = {}
    self.morphology['length'] = nw.data_dict['lengths']
    # each item in this sub-dictionary is the per-frame mean across some
    # part of the worm the head, midbody and tail.
    #
    # shape of resulting arrays are (2, n)
    width_dict = {k: np.mean(nw.get_partition(k, 'skeletons'), 0) \
                  for k in ('head', 'midbody', 'tail')}
    self.morphology['width'] = width_dict
    self.morphology['area'] = nw.data_dict['head_areas'] + \
                              nw.data_dict['vulva_areas'] + \
                              nw.data_dict['non_vulva_areas']
    self.morphology['areaPerLength'] = self.morphology['area'] / \
                                       self.morphology['length']
    self.morphology['widthPerLength'] = self.morphology['width']['midbody'] / \
                                        self.morphology['length']
    
  def num_fields(self):
    return 
  
  def get_posture_features(self):
    """
    Translation of: SegwormMatlabClasses / 
    +seg_worm / @feature_calculator / getPostureFeatures.m

    %
    %   posture = seg_worm.feature_calculator.getPostureFeatures(nw)
    %
    %   Old Files
    %   - schaferFeatures_process
    %
    %   NOTES:
    %   - Indices were inconsistently defined for bends relative to other code
    %   - stdDev for bends is signed as well, based on means ...
    %
    %   UNFINISHED STATUS:
    %   - seg_worm.feature_helpers.posture.wormKinks - not yet examined
    %   - distance - missing input to function, need to process locomotion
    %   first
    %

    """    
    nw = self.normalized_worm # let's use this convenient alias 
    
    # Initialize self.posture as a blank dictionary we will add to
    self.posture = {}  

    # *** 1. Bends ***

    # Now that we've populated the bends dictionary, add it to the posture
    # dictionary.
    self.posture['bends'] = feature_helpers.get_bends(nw)
    
    # *** 2. Eccentricity & Orientation ***
    eccentricity_and_orientation = \
            feature_helpers.get_eccentricity_and_orientation(nw.contour_x(), 
                                                             nw.contour_y())
    self.posture['eccentricity'] = eccentricity_and_orientation.eccentricity
    self.posture['orientation'] = eccentricity_and_orientation.orientation
    
    # *** 3. Amplitude, Wavelengths, TrackLength, Amplitude Ratio ***
    amp_wave_track = \
      collections.namedtuple('amp_wave_track', 
                             ['amplitude', 'wavelength', 'track_length'])
    amp_wave_track.amplitude = 'yay1'
    amp_wave_track.wavelength = 'yay2'
    amp_wave_track.track_length = 'yay3'

    #amp_wave_track = get_amplitude_and_wavelength( \
    #                      self.posture['orientation'],
    #                      self.skeletons_x(),
    #                      self.skeletons_y(),
    #                      self.data_dict['lengths'])
    self.posture['amplitude'] = amp_wave_track.amplitude
    self.posture['wavelength'] = amp_wave_track.wavelength
    self.posture['track_length'] = amp_wave_track.track_length

    # TODO: change this to return multiple values as in 
    # http://stackoverflow.com/questions/354883/how-do-you-return-multiple-values-in-python

    # *** 4. Kinks ***
    

    # *** 5. Coils ***

    # *** 6. Directions ***

    # *** 7. Skeleton ***  
    # (already in morphology, but Schafer Lab put it here too)

    
    # *** 8. EigenProjection ***
    
    pass

  
  def get_locomotion_features(self):
    """
    Translation of: SegwormMatlabClasses / 
    +seg_worm / @feature_calculator / getLocomotionFeatures.m

    """
    locomotion = {}
    
    # until we have the below calculated, just create an array of zeroes
    locomotion['midbody_distance'] = 0
    #      np.zeros(np.shape(self.skeletons_x()))        
    #      abs(locomotion['velocity']['midbody']['speed'] / config.FPS)
    
    pass
 
  
  def get_path_features(self):
    """
    Translation of: SegwormMatlabClasses / 
    +seg_worm / @feature_calculator / getPathFeatures.m

    """    
    pass





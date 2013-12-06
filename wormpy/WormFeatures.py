# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 22:36:32 2013

@author: Michael
A translation of Matlab code written by Jim Hokanson,
in the SegwormMatlabClasses GitHub repo.  Original code path:
SegwormMatlabClasses / 
+seg_worm / @feature_calculator / get_features_rewritten.m
"""
import numpy as np

class WormFeatures:
  """ WormFeatures takes as input a NormalizedWorm instance, and
      during initialization calculates all the features of the worm.
  """
  normalized_worm = None
  
  morphology = None  # a python dictionary
  locomotion = None
  posture = None
  path = None
  
  def __init__(self, nw):
    self.normalized_worm = nw
    self.get_morphology_features()
    self.get_locomotion_features()
    self.get_posture_features()

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
    width_dict = {k: np.mean(nw.get_partition(k), 0) \
                  for k in ('head', 'midbody', 'tail')}
    self.morphology['width'] = width_dict
    self.morphology['area'] = nw.data_dict['head_areas'] + \
                              nw.data_dict['vulva_areas'] + \
                              nw.data_dict['non_vulva_areas']
    self.morphology['areaPerLength'] = self.morphology['area'] / \
                                       self.morphology['length']
    self.morphology['widthPerLength'] = self.morphology['width']['midbody'] / \
                                        self.morphology['length']
    
  
  def get_locomotion_features(self):
    """
    Translation of: SegwormMatlabClasses / 
    +seg_worm / @feature_calculator / getLocomotionFeatures.m

    """
    pass
  
  def get_posture_features(self):
    """
    Translation of: SegwormMatlabClasses / 
    +seg_worm / @feature_calculator / getPostureFeatures.m

    """    
    pass
  
  def get_path_features(self):
    """
    Translation of: SegwormMatlabClasses / 
    +seg_worm / @feature_calculator / getPathFeatures.m

    """    
    pass

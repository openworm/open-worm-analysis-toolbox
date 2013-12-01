# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 22:36:32 2013

@author: Michael
A translation of Matlab code written by Jim Hokanson,
in the SegwormMatlabClasses GitHub repo.  Original code path:
SegwormMatlabClasses / 
+seg_worm / @feature_calculator / get_features_rewritten.m
"""

class WormFeatures:
  """ WormFeatures takes as input a NormalizedWorm instance, and
      during initialization calculates all the features of the worm.
  """
  normalized_worm = None
  
  morphology = None
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
    
    Inputs: normalized_worm
    Outputs: a structure called "morphology"
    %Old files that served as a reference ...
    %------------------------------------------------------------
    %morphology_process.m
    %schaferFeatures_process.m
    SI = seg_worm.skeleton_indices;
        morphology.length         = nw.lengths;
    morphology.width.head     = mean(nw.widths(SI.HEAD_INDICES,:),1);
    morphology.width.midbody  = mean(nw.widths(SI.MID_INDICES,:),1);
    morphology.width.tail     = mean(nw.widths(SI.TAIL_INDICES,:),1);
    morphology.area           = nw.head_areas + nw.tail_areas + nw.vulva_areas + nw.non_vulva_areas;
    morphology.areaPerLength  = morphology.area./morphology.length;
    morphology.widthPerLength = morphology.width.midbody./morphology.length;
    """
    pass
  
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